#!/usr/bin/env python3
r"""
nested_parser.py —— 容错的嵌套结构解析器（纯标准库）。

功能：
  * 识别圆括号 ()、方括号 []、花括号 {} 表示的嵌套结构；
  * 单/双引号字符串、# 行注释、// 行注释、/* */ 块注释中的括号不算语法；
  * 遇到多闭 / 多开 / 类型不匹配时不中断，按既定策略恢复并收集全部错误；
  * 嵌套深度超过上限时报错并恢复；
  * 树构建完成后对叶子节点（裸原子）做内容格式校验；
  * 输出结构树与完整错误清单。

用法：
  python3 nested_parser.py input.txt
  cat input.txt | python3 nested_parser.py -
  python3 nested_parser.py input.txt --max-depth 32 \
      --leaf-pattern "[A-Za-z_][A-Za-z0-9_]*|-?\d+(\.\d+)?"
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

OPEN_TO_CLOSE = {"(": ")", "[": "]", "{": "}"}
CLOSE_TO_OPEN = {v: k for k, v in OPEN_TO_CLOSE.items()}

# 默认允许的裸叶子格式：标识符 或 整数/小数（引号字符串不做此校验）
DEFAULT_LEAF_PATTERN = r"[A-Za-z_][A-Za-z0-9_]*|-?\d+(\.\d+)?"


@dataclass
class ParseError:
    """一条诊断信息。expected 给出“本应出现的括号”，便于定位。"""

    line: int
    col: int
    kind: str  # unclosed | extra_close | mismatch | depth | bad_string | bad_comment | bad_leaf
    message: str
    expected: Optional[str] = None

    def __str__(self) -> str:
        tail = f"；期望 {self.expected!r}" if self.expected is not None else ""
        return f"第{self.line}行第{self.col}列 [{self.kind}] {self.message}{tail}"


@dataclass
class Node:
    """结构树节点。kind 为 list（括号节点/根）或 atom（叶子）。"""

    kind: str
    line: int = 0
    col: int = 0
    bracket: Optional[str] = None  # list 节点的开括号；虚拟根为 None
    value: Optional[str] = None  # atom 的文本内容（不含外层引号）
    is_string: bool = False  # atom 是否来自引号字符串
    children: List["Node"] = field(default_factory=list)


@dataclass
class Token:
    type: str  # open | close | atom
    value: str
    line: int
    col: int
    is_string: bool = False


# --------------------------------------------------------------------------- #
# 词法分析：把文本切成 括号/叶子 记号，同时剔除字符串与注释中的括号。
# --------------------------------------------------------------------------- #
def tokenize(text: str, errors: List[ParseError]) -> List[Token]:
    tokens: List[Token] = []
    buf: List[str] = []
    buf_line = buf_col = 0

    def flush() -> None:
        nonlocal buf, buf_line, buf_col
        if buf:
            tokens.append(Token("atom", "".join(buf), buf_line, buf_col))
            buf = []

    i, n = 0, len(text)
    line, col = 1, 1

    def advance() -> str:
        nonlocal i, line, col
        ch = text[i]
        i += 1
        if ch == "\n":
            line += 1
            col = 1
        else:
            col += 1
        return ch

    while i < n:
        ch = text[i]

        # 空白与逗号（逗号仅作分隔符）
        if ch in " \t\r\n,":
            flush()
            advance()
            continue

        # 括号
        if ch in OPEN_TO_CLOSE or ch in CLOSE_TO_OPEN:
            flush()
            kind = "open" if ch in OPEN_TO_CLOSE else "close"
            tokens.append(Token(kind, ch, line, col))
            advance()
            continue

        # 字符串（括号不视为语法，\ 可转义下一字符）
        if ch in "\"'":
            flush()
            start_line, start_col, quote = line, col, ch
            advance()
            chars: List[str] = []
            closed = False
            while i < n:
                c = text[i]
                if c == "\\":
                    advance()
                    if i < n:  # 保留转义对的原始内容
                        chars.append("\\" + text[i])
                        advance()
                    continue
                if c == quote:
                    closed = True
                    advance()
                    break
                if c == "\n":  # 单行字符串不允许跨行，报错并在行尾恢复
                    break
                chars.append(advance())
            if not closed:
                errors.append(
                    ParseError(
                        start_line,
                        start_col,
                        "bad_string",
                        f"未闭合的字符串 {quote}（已在行尾恢复）",
                        expected=quote,
                    )
                )
            tokens.append(Token("atom", "".join(chars), start_line, start_col, is_string=True))
            continue

        # 行注释 # 与 //
        if ch == "#" or (ch == "/" and i + 1 < n and text[i + 1] == "/"):
            flush()
            length = 2 if ch == "/" else 1
            for _ in range(length):
                advance()
            while i < n and text[i] != "\n":
                advance()
            continue

        # 块注释 /* ... */（可跨行，记录起始行列以便报错）
        if ch == "/" and i + 1 < n and text[i + 1] == "*":
            flush()
            start_line, start_col = line, col
            advance()
            advance()
            closed = False
            while i < n:
                if text[i] == "*" and i + 1 < n and text[i + 1] == "/":
                    advance()
                    advance()
                    closed = True
                    break
                advance()
            if not closed:
                errors.append(
                    ParseError(
                        start_line,
                        start_col,
                        "bad_comment",
                        "未闭合的块注释 /*（已在文件尾恢复）",
                        expected="*/",
                    )
                )
            continue

        # 普通字符累积为裸原子
        if not buf:
            buf_line, buf_col = line, col
        buf.append(advance())

    flush()
    return tokens


# --------------------------------------------------------------------------- #
# 语法分析：栈式建树 + 容错恢复。
#
# 恢复策略（错误不阻断解析，全部收集）：
#   1. 多闭（栈空仍出现闭括号，或闭括号不属于任何外层）—— 跳过该闭括号。
#      理由：它没有可以配对的开括号，补“虚拟开括号”会凭空制造节点；
#            跳过对后续结构破坏最小。
#   2. 多开 / EOF 未闭合 —— 对栈中剩余开括号统一“补闭”。
#      理由：已读到的子树是确定的，补闭只是收口，不丢失任何已解析内容。
#   3. 类型不匹配（闭括号与栈顶不符）—— 若它匹配某个外层开括号，则认为
#      中间各层“漏写闭括号”，逐层补闭并收口到匹配层；否则按“多闭”跳过。
#      理由：如 ( [ )，真实意图通常是 [ 漏了 ]，补闭比凭空丢节点合理。
#   4. 超过深度上限 —— 该开括号不下压（忽略），继续在当前层解析；
#      其将来的配闭会按“多闭”跳过并单独报错，边界行为可预测。
# --------------------------------------------------------------------------- #
def build_tree(
    tokens: List[Token], errors: List[ParseError], max_depth: int
) -> Node:
    root = Node("list", line=1, col=1, bracket=None)
    stack: List[Node] = [root]

    for tok in tokens:
        if tok.type == "atom":
            stack[-1].children.append(
                Node("atom", tok.line, tok.col, value=tok.value, is_string=tok.is_string)
            )
            continue

        if tok.type == "open":
            new_depth = len(stack)  # 新节点下压后的深度（根为 0）
            if new_depth > max_depth:
                errors.append(
                    ParseError(
                        tok.line,
                        tok.col,
                        "depth",
                        f"嵌套深度超过上限 {max_depth}，已忽略该开括号 {tok.value}",
                        expected=None,
                    )
                )
                continue
            node = Node("list", tok.line, tok.col, bracket=tok.value)
            stack[-1].children.append(node)
            stack.append(node)
            continue

        # close
        wanted = (
            OPEN_TO_CLOSE[stack[-1].bracket]
            if stack[-1].bracket is not None
            else None
        )

        # 在栈中（不含虚拟根）自内向外找同类型开括号
        match_idx = -1
        for idx in range(len(stack) - 1, 0, -1):
            if OPEN_TO_CLOSE[stack[idx].bracket] == tok.value:
                match_idx = idx
                break

        if match_idx == -1:
            # 多闭：无法与任何外层配对 —— 跳过
            errors.append(
                ParseError(
                    tok.line,
                    tok.col,
                    "extra_close",
                    f"多余的闭括号 {tok.value}（没有对应的开括号，已跳过）",
                    expected=wanted,
                )
            )
            continue

        if match_idx == len(stack) - 1:
            stack.pop()  # 正常配对
            continue

        # 类型不匹配：为中间层补闭，收口到匹配的外层
        missing = stack[match_idx + 1 :]
        missing_desc = "、".join(
            f"第{nd.line}行的 {nd.bracket}{OPEN_TO_CLOSE[nd.bracket]}"
            for nd in missing
        )
        errors.append(
            ParseError(
                tok.line,
                tok.col,
                "mismatch",
                (
                    f"闭括号 {tok.value} 与栈顶不匹配；已为漏写的闭括号自动补闭"
                    f"（{missing_desc}），并用 {tok.value} 收口"
                ),
                expected=wanted,
            )
        )
        del stack[match_idx:]

    # EOF：栈中剩余开括号一律补闭（多开恢复）
    for frame in stack[1:]:
        errors.append(
            ParseError(
                frame.line,
                frame.col,
                "unclosed",
                f"开括号 {frame.bracket} 到文件尾仍未闭合（已自动补闭）",
                expected=OPEN_TO_CLOSE[frame.bracket],
            )
        )

    return root


# --------------------------------------------------------------------------- #
# 叶子格式校验：树构建完成后再做（语法正确性与内容合法性分开报告）。
# --------------------------------------------------------------------------- #
def validate_leaves(root: Node, leaf_pattern: str, errors: List[ParseError]) -> None:
    pattern = re.compile(leaf_pattern)

    def walk(node: Node) -> None:
        if node.kind == "atom":
            if node.is_string:
                return  # 引号字符串内容不做格式限制
            if not pattern.fullmatch(node.value or ""):
                errors.append(
                    ParseError(
                        node.line,
                        node.col,
                        "bad_leaf",
                        f"叶子内容 {node.value!r} 不符合格式 {leaf_pattern}",
                    )
                )
            return
        for child in node.children:
            walk(child)

    walk(root)


def parse(
    text: str, max_depth: int = 64, leaf_pattern: str = DEFAULT_LEAF_PATTERN
) -> Tuple[Node, List[ParseError]]:
    """主入口：返回 结构树根节点 与 错误清单。"""
    errors: List[ParseError] = []
    tokens = tokenize(text, errors)
    root = build_tree(tokens, errors, max_depth)
    validate_leaves(root, leaf_pattern, errors)
    errors.sort(key=lambda e: (e.line, e.col))
    return root, errors


def format_tree(node: Node, indent: int = 0) -> List[str]:
    pad = "  " * indent
    lines: List[str] = []
    if node.kind == "atom":
        shown = f'"{node.value}"' if node.is_string else node.value
        lines.append(f"{pad}{shown}")
        return lines

    if node.bracket is None:
        lines.append(f"{pad}<root>  (第1行)")
    else:
        lines.append(f"{pad}{node.bracket}  (第{node.line}行开)")
    for child in node.children:
        lines.extend(format_tree(child, indent + 1))
    if node.bracket is not None:
        lines.append(f"{pad}{OPEN_TO_CLOSE[node.bracket]}")
    return lines


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="容错嵌套结构解析器")
    ap.add_argument("file", help="输入文件；- 表示标准输入")
    ap.add_argument("--max-depth", type=int, default=64, help="最大嵌套深度，默认 64")
    ap.add_argument(
        "--leaf-pattern",
        default=DEFAULT_LEAF_PATTERN,
        help="裸叶子允许的正则（re.fullmatch），默认标识符或数字",
    )
    args = ap.parse_args(argv)

    if args.file == "-":
        text = sys.stdin.read()
    else:
        with open(args.file, "r", encoding="utf-8") as fh:
            text = fh.read()

    root, errors = parse(text, max_depth=args.max_depth, leaf_pattern=args.leaf_pattern)

    print("=== 结构树 ===")
    print("\n".join(format_tree(root)))
    print()
    print(f"=== 错误清单（共 {len(errors)} 条）===")
    for err in errors:
        print(f"- {err}")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
