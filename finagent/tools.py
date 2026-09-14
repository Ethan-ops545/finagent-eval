# -*- coding: utf-8 -*-
"""Agent 可用的工具。

五个工具，分两类：

  取数类（list_periods / get_income_statement / get_balance_sheet / get_market_data）
      返回数据的同时，把"本次让 Agent 看到了哪些字段"记进 trace。

  计算类（compute）
      Agent **不允许自己心算**。它只能提交「公式 + 每个变量对应的字段引用」，
      由工具回到快照取原始值、用受限求值器算出来。
      这么设计的直接后果：算术错误和取数错误被彻底分开了——
      compute 的结果永远等于"这个公式配这组输入"的正确答案，
      Agent 还能犯的错就只剩「选错公式」和「选错输入」，而这两种评测器都查得出来。

关于前视偏差：取数工具**不会**替 Agent 挡住晚于 as_of 发布的数据。
这是有意的——真实系统里数据库并不知道你的研究基准日，
把前视做成"工具层封死"会让评测检不出这类错误，而现实中它恰恰是最常见的。
所以这里让它可能发生，再由评测器抓出来。
"""
from __future__ import annotations

import time
from typing import Any, Callable

from .calc import safe_eval, CalcError
from .snapshot import Snapshots, RefError
from .trace import ToolCall, Trace


class ToolError(RuntimeError):
    pass


class Toolbox:
    def __init__(self, snaps: Snapshots, trace: Trace):
        self.snaps = snaps
        self.trace = trace
        self._step = 0

    # ---------- 具体工具 ----------

    def list_periods(self, ticker: str) -> dict[str, Any]:
        snap = self.snaps.load(ticker)
        return {
            "ticker": snap["ticker"],
            "name": snap["name"],
            "sector": snap["sector"],
            "income_periods": sorted(snap["statements"]["income"]),
            "balance_periods": sorted(snap["statements"]["balance"]),
            "market_dates": sorted(snap["market"]),
        }, []

    def _statement(self, kind: str, ticker: str, period: str):
        fields = self.snaps.public_fields(ticker, kind, period)
        block = self.snaps.period_block(ticker, kind, period)
        refs = [f"{ticker.upper()}.{kind}.{period}.{k}" for k in fields]
        return {
            "ticker": ticker.upper(),
            "statement": kind,
            "period": period,
            "published": block.get("_published"),
            "unit": block.get("_unit"),
            "fields": fields,
        }, refs

    def get_income_statement(self, ticker: str, period: str):
        return self._statement("income", ticker, period)

    def get_balance_sheet(self, ticker: str, period: str):
        return self._statement("balance", ticker, period)

    def get_market_data(self, ticker: str, date: str):
        return self._statement("market", ticker, date)

    def compute(self, expression: str, inputs: dict[str, str]):
        """按「公式 + 字段引用」做确定性计算。inputs: 变量名 -> 字段引用。"""
        if not isinstance(inputs, dict) or not inputs:
            raise ToolError("inputs 必须是「变量名 -> 字段引用」的非空字典")
        resolved, values = {}, {}
        for var, ref in inputs.items():
            r = self.snaps.resolve(ref)
            values[var] = r.value
            resolved[var] = {"ref": ref, "value": r.value, "published": r.published}
        value = safe_eval(expression, values)
        return {
            "expression": expression,
            "value": value,
            "resolved": resolved,
        }, list(inputs.values())

    # ---------- 分发与追踪 ----------

    HANDLERS: dict[str, str] = {
        "list_periods": "list_periods",
        "get_income_statement": "get_income_statement",
        "get_balance_sheet": "get_balance_sheet",
        "get_market_data": "get_market_data",
        "compute": "compute",
    }

    def call(self, name: str, args: dict) -> dict:
        self._step += 1
        t0 = time.perf_counter()
        if name not in self.HANDLERS:
            self.trace.add(ToolCall(self._step, name, args, False,
                                    (time.perf_counter() - t0) * 1000,
                                    error=f"未知工具: {name}"))
            return {"error": f"未知工具: {name}"}
        fn: Callable = getattr(self, self.HANDLERS[name])
        try:
            result, refs = fn(**args)
        except (RefError, CalcError, ToolError, TypeError, KeyError) as e:
            msg = f"{type(e).__name__}: {e}"
            self.trace.add(ToolCall(self._step, name, args, False,
                                    (time.perf_counter() - t0) * 1000, error=msg))
            return {"error": msg}
        self.trace.add(ToolCall(self._step, name, args, True,
                                (time.perf_counter() - t0) * 1000,
                                exposed_refs=refs))
        return result


# ---------- 给大模型看的工具声明（Anthropic tool-use 格式） ----------

TOOL_SPECS: list[dict] = [
    {
        "name": "list_periods",
        "description": "列出某公司可用的报表期间与市场数据日期。开始分析前先调用它。",
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string", "description": "公司代码，如 NOVA"}},
            "required": ["ticker"],
        },
    },
    {
        "name": "get_income_statement",
        "description": "取某公司某一期的利润表。返回值里的 published 是该期财报的发布日。",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "period": {"type": "string", "description": "如 FY2024"},
            },
            "required": ["ticker", "period"],
        },
    },
    {
        "name": "get_balance_sheet",
        "description": "取某公司某一期的资产负债表。",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "period": {"type": "string", "description": "如 FY2024"},
            },
            "required": ["ticker", "period"],
        },
    },
    {
        "name": "get_market_data",
        "description": "取某公司某一日的收盘价与在外股数。",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "date": {"type": "string", "description": "如 2025-04-15"},
            },
            "required": ["ticker", "date"],
        },
    },
    {
        "name": "compute",
        "description": (
            "做算术。你不得自己心算任何数值——所有计算必须经过本工具。"
            "提交一个公式和每个变量对应的字段引用（格式 TICKER.statement.PERIOD.field），"
            "工具会回到原始数据取值并计算。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string",
                               "description": "只含变量名、数字与 + - * / ** ( ) 的表达式，如 rev_cur / rev_prev - 1"},
                "inputs": {"type": "object",
                           "description": "变量名 -> 字段引用，如 {\"rev_cur\": \"NOVA.income.FY2024.revenue\"}",
                           "additionalProperties": {"type": "string"}},
            },
            "required": ["expression", "inputs"],
        },
    },
]
