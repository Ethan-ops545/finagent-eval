# -*- coding: utf-8 -*-
"""受限表达式求值器。

Agent 的任何算术都必须经过这里，评测器复算时也用同一个求值器——
这样"算错"就只可能来自 Agent 选错了公式或选错了输入，
而不可能来自两边的计算实现不一致。

刻意**不用 `eval()`**：Agent 产出的公式字符串属于不可信输入，
直接 eval 等于把任意代码执行权交给大模型（JD 里说的 Prompt 注入面之一）。
这里走 AST 白名单：只允许数字、四则运算、幂、一元正负号，以及事先绑定好的变量名。
"""
from __future__ import annotations

import ast
import operator

_BIN = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UN = {ast.UAdd: operator.pos, ast.USub: operator.neg}


class CalcError(ValueError):
    """公式非法、变量未绑定或除零。"""


def safe_eval(expr: str, variables: dict[str, float]) -> float:
    """按白名单求值 `expr`，变量取自 `variables`。"""
    if not isinstance(expr, str) or not expr.strip():
        raise CalcError("空表达式")
    if len(expr) > 500:
        raise CalcError("表达式过长")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise CalcError(f"语法错误: {e}") from e

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise CalcError(f"不允许的常量: {node.value!r}")
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            right = ev(node.right)
            if isinstance(node.op, ast.Div) and right == 0:
                raise CalcError("除零")
            return _BIN[type(node.op)](ev(node.left), right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UN:
            return _UN[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Name):
            if node.id not in variables:
                raise CalcError(f"未绑定的变量: {node.id}")
            return float(variables[node.id])
        raise CalcError(f"不允许的表达式节点: {type(node).__name__}")

    return float(ev(tree))
