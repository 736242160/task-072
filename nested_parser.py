#!/usr/bin/env python3
"""fault-tolerant nested-structure parser (stdlib only).

Parses text with (), [], {} nesting, strings and comments, recovers from
bracket errors, collects all errors, and prints the structure tree plus
the full error list.

Usage:
    python3 nested_parser.py [FILE] [--max-depth N] [--json]
    (reads stdin when FILE is omitted or "-")
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field

OPEN_TO_CLOSE = {"(": ")", "[": "]", "{": "}"}
CLOSE_TO_OPEN = {v: k for k, v in OPEN_TO_CLOSE.items()}

# Leaf content formats: number, or identifier-like atom.
RE_NUMBER = re.compile(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?")
RE_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_.\-]*")


# ---------------------------------------------------------------- errors
@dataclass
class ParseError:
    line: int
    col: int
    kind: str
    message: str

    def __str__(self):
        return f"line {self.line}, col {self.col}: [{self.kind}] {self.message}"


# ---------------------------------------------------------------- tokens
@dataclass
class Token:
    kind: str   # 'OPEN' | 'CLOSE' | 'ATOM' | 'STRING'
    value: str
    line: int
    col: int


def tokenize(text, errors):
    """Scan text into tokens. Brackets inside strings/comments are inert."""
    tokens = []
    i, line, col = 0, 1, 1
    n = len(text)

    def advance(ch):
        nonlocal line, col
        if ch == "\n":
            line += 1
            col = 1
        else:
            col += 1

    while i < n:
        ch = text[i]

        if ch.isspace():
            advance(ch)
            i += 1
            continue

        # comments: # ... or // ... to end of line, /* ... */ block
        if ch == "#" or (ch == "/" and i + 1 < n and text[i + 1] == "/"):
            while i < n and text[i] != "\n":
                advance(text[i])
                i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "*":
            start_line, start_col = line, col
            advance(text[i]); advance(text[i + 1])
            i += 2
            closed = False
            while i < n:
                if text[i] == "*" and i + 1 < n and text[i + 1] == "/":
                    advance(text[i]); advance(text[i + 1])
                    i += 2
                    closed = True
                    break
                advance(text[i])
                i += 1
            if not closed:
                errors.append(ParseError(
                    start_line, start_col, "UnterminatedComment",
                    "block comment opened here is never closed"))
            continue

        # strings: '...' or "..." with backslash escapes
        if ch in "\"'":
            quote = ch
            start_line, start_col = line, col
            advance(ch)
            i += 1
            buf = []
            closed = False
            while i < n:
                c = text[i]
                if c == "\\" and i + 1 < n:
                    buf.append(text[i + 1])
                    advance(text[i]); advance(text[i + 1])
                    i += 2
                    continue
                if c == quote:
                    advance(c)
                    i += 1
                    closed = True
                    break
                if c == "\n":
                    break  # unterminated: stop at end of line
                buf.append(c)
                advance(c)
                i += 1
            if not closed:
                errors.append(ParseError(
                    start_line, start_col, "UnterminatedString",
                    f"string opened with {quote!r} here is never closed; "
                    "recovered by ending it at end of line"))
            tokens.append(Token("STRING", "".join(buf), start_line, start_col))
            continue

        if ch in OPEN_TO_CLOSE:
            tokens.append(Token("OPEN", ch, line, col))
            advance(ch)
            i += 1
            continue
        if ch in CLOSE_TO_OPEN:
            tokens.append(Token("CLOSE", ch, line, col))
            advance(ch)
            i += 1
            continue

        # atom: run of non-delimiter characters (leaf content)
        start_line, start_col = line, col
        buf = []
        while i < n and not text[i].isspace() and text[i] not in "()[]{}\"'#":
            if text[i] == "/" and i + 1 < n and text[i + 1] in "/*":
                break
            buf.append(text[i])
            advance(text[i])
            i += 1
        tokens.append(Token("ATOM", "".join(buf), start_line, start_col))

    return tokens


# ---------------------------------------------------------------- nodes
@dataclass
class Node:
    kind: str                  # 'list' | 'leaf'
    open_ch: str = ""          # for lists: ( [ {
    children: list = field(default_factory=list)
    value: str = ""            # for leaves
    leaf_type: str = ""        # 'atom' | 'string'
    line: int = 0
    col: int = 0
    synthetic_close: bool = False  # close bracket was inserted by recovery

    def to_obj(self):
        if self.kind == "leaf":
            return {"leaf": self.value, "type": self.leaf_type,
                    "line": self.line}
        return {"list": self.open_ch,
                "children": [c.to_obj() for c in self.children],
                "line": self.line,
                "auto_closed": self.synthetic_close}


# ---------------------------------------------------------------- parser
class Parser:
    """Stack-based parser with error recovery.

    Recovery strategy:
      * extra close bracket (no matching open anywhere on the stack):
        SKIP the close token -- it pairs with nothing, so dropping it
        disturbs the surrounding structure the least.
      * mismatched close, but a matching open exists deeper in the stack:
        AUTO-CLOSE the intervening opens (report each) and accept the
        close -- the close bracket is strong evidence of intent, while
        the unclosed inner opens are likely the mistake.
      * opens still pending at EOF: AUTO-CLOSE them (report each) --
        the only way to still deliver a complete tree.
    """

    def __init__(self, max_depth=100):
        self.max_depth = max_depth
        self.errors = []
        self.roots = []
        self.stack = []  # frames: Node('list') currently being built

    def _error(self, line, col, kind, message):
        self.errors.append(ParseError(line, col, kind, message))

    def _close_frame(self, frame, synthetic):
        frame.synthetic_close = synthetic
        if self.stack:
            self.stack[-1].children.append(frame)
        else:
            self.roots.append(frame)

    def parse(self, tokens):
        for tok in tokens:
            if tok.kind == "OPEN":
                depth = len(self.stack) + 1
                if depth > self.max_depth:
                    self._error(tok.line, tok.col, "MaxDepthExceeded",
                                f"nesting depth {depth} exceeds limit "
                                f"{self.max_depth} at {tok.value!r}; "
                                "node kept but flagged")
                node = Node("list", open_ch=tok.value,
                            line=tok.line, col=tok.col)
                self.stack.append(node)
            elif tok.kind in ("ATOM", "STRING"):
                leaf = Node("leaf", value=tok.value,
                            leaf_type="atom" if tok.kind == "ATOM" else "string",
                            line=tok.line, col=tok.col)
                if self.stack:
                    self.stack[-1].children.append(leaf)
                else:
                    self.roots.append(leaf)
            elif tok.kind == "CLOSE":
                self._handle_close(tok)

        # EOF: auto-close any opens still pending (recovery for 多开).
        while self.stack:
            frame = self.stack.pop()
            self._error(frame.line, frame.col, "UnclosedOpen",
                        f"open bracket {frame.open_ch!r} is never closed; "
                        f"auto-inserted {OPEN_TO_CLOSE[frame.open_ch]!r} at EOF")
            self._close_frame(frame, synthetic=True)
        return self.roots

    def _handle_close(self, tok):
        expected = OPEN_TO_CLOSE[self.stack[-1].open_ch] if self.stack else None

        if not self.stack:
            self._error(tok.line, tok.col, "UnexpectedClose",
                        f"close bracket {tok.value!r} has no matching open; "
                        "skipped it and continued")
            return

        if tok.value == expected:
            self._close_frame(self.stack.pop(), synthetic=False)
            return

        # Mismatch. Is there a matching open deeper in the stack?
        want_open = CLOSE_TO_OPEN[tok.value]
        match_idx = None
        for idx in range(len(self.stack) - 1, -1, -1):
            if self.stack[idx].open_ch == want_open:
                match_idx = idx
                break

        if match_idx is None:
            self._error(tok.line, tok.col, "UnexpectedClose",
                        f"close bracket {tok.value!r} matches nothing "
                        f"(innermost open expects {expected!r}); "
                        "skipped it and continued")
            return

        # Auto-close everything above the matching open, then accept.
        self._error(tok.line, tok.col, "MismatchedClose",
                    f"expected {expected!r} but found {tok.value!r}; "
                    f"auto-closing {len(self.stack) - 1 - match_idx} inner "
                    "open(s) to accept it")
        while len(self.stack) - 1 > match_idx:
            frame = self.stack.pop()
            self._error(frame.line, frame.col, "UnclosedOpen",
                        f"open bracket {frame.open_ch!r} auto-closed with "
                        f"{OPEN_TO_CLOSE[frame.open_ch]!r} to recover")
            self._close_frame(frame, synthetic=True)
        self._close_frame(self.stack.pop(), synthetic=False)


# ---------------------------------------------------------------- leaf validation
def validate_leaves(nodes, errors):
    """After the tree is built, check every leaf's content format."""
    for node in nodes:
        if node.kind == "leaf":
            if node.leaf_type == "string":
                continue  # any content is legal inside quotes
            v = node.value
            if not v:
                errors.append(ParseError(node.line, node.col, "InvalidLeaf",
                                         "empty leaf content"))
            elif not (RE_NUMBER.fullmatch(v) or RE_IDENT.fullmatch(v)):
                errors.append(ParseError(
                    node.line, node.col, "InvalidLeaf",
                    f"leaf {v!r} is neither a number nor an identifier "
                    "(allowed: letters, digits, '_', '-', '.')"))
        else:
            validate_leaves(node.children, errors)


# ---------------------------------------------------------------- output
def render_tree(nodes, indent=0):
    pad = "  " * indent
    lines = []
    for node in nodes:
        if node.kind == "leaf":
            tag = "str" if node.leaf_type == "string" else "atom"
            lines.append(f"{pad}{tag}: {node.value!r}  (line {node.line})")
        else:
            close = OPEN_TO_CLOSE[node.open_ch]
            mark = "  [auto-closed]" if node.synthetic_close else ""
            lines.append(f"{pad}{node.open_ch} (line {node.line}){mark}")
            lines.extend(render_tree(node.children, indent + 1))
            lines.append(f"{pad}{close}")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description="fault-tolerant nested-structure parser")
    ap.add_argument("file", nargs="?", default="-",
                    help="input file (default: stdin)")
    ap.add_argument("--max-depth", type=int, default=100,
                    help="maximum nesting depth (default: 100)")
    ap.add_argument("--json", action="store_true",
                    help="emit the tree as JSON")
    args = ap.parse_args(argv)

    text = sys.stdin.read() if args.file == "-" else open(args.file, encoding="utf-8").read()

    lex_errors = []
    tokens = tokenize(text, lex_errors)

    parser = Parser(max_depth=args.max_depth)
    roots = parser.parse(tokens)

    all_errors = lex_errors + parser.errors
    validate_leaves(roots, all_errors)
    all_errors.sort(key=lambda e: (e.line, e.col))

    print("=== structure tree ===")
    if args.json:
        print(json.dumps([r.to_obj() for r in roots], ensure_ascii=False, indent=2))
    else:
        for line in render_tree(roots):
            print(line)

    print("\n=== errors (%d) ===" % len(all_errors))
    if all_errors:
        for err in all_errors:
            print(err)
    else:
        print("(none)")
    return 1 if all_errors else 0


if __name__ == "__main__":
    sys.exit(main())
