# -*- coding: utf-8 -*-
"""Agent 输出契约。

这个文件是整个项目的核心设计。Agent **不允许**只输出一段自然语言，
它必须把结论拆成一条条 Claim，并声明每条属于哪一类：

  computed  —— 算出来的。必须给出 formula 与 inputs，评测器会独立复算。
  retrieved —— 直接从数据里取的。必须给出唯一的 inputs，评测器会回原始数据比对。
  judgment  —— 判断性表述（"增长稳健""盈利质量改善"）。无法复算，走 rubric 打分。

这条分界线就是 JD 里说的"通过 LLM 或确定性逻辑兜底实现"：
凡是能复算的，一律不信模型的话，回到代码复算；只有真正需要判断的才交给模型。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any

KINDS = ("computed", "retrieved", "judgment")


class ContractError(ValueError):
    """Agent 的输出不符合契约。"""


@dataclass
class Claim:
    id: str
    kind: str                                   # computed / retrieved / judgment
    text: str                                   # 给人读的一句话
    value: float | None = None
    unit: str | None = None                     # ratio / multiple / 千美元 / …
    formula: str | None = None                  # 仅 computed
    inputs: dict[str, str] = field(default_factory=dict)   # 变量名 -> 字段引用
    metric: str | None = None                   # 对应 benchmark 里的 metric 名

    def validate(self) -> None:
        if self.kind not in KINDS:
            raise ContractError(f"[{self.id}] 未知的 kind: {self.kind}")
        if self.kind == "judgment":
            if self.value is not None:
                raise ContractError(f"[{self.id}] judgment 类结论不应带数值")
            return
        if self.value is None:
            raise ContractError(f"[{self.id}] {self.kind} 类结论必须带数值")
        if not self.inputs:
            raise ContractError(f"[{self.id}] {self.kind} 类结论必须声明 inputs（证据地址）")
        if self.kind == "computed" and not self.formula:
            raise ContractError(f"[{self.id}] computed 类结论必须给出 formula")
        if self.kind == "retrieved" and len(self.inputs) != 1:
            raise ContractError(f"[{self.id}] retrieved 类结论应恰好有一个 input")


@dataclass
class Report:
    task_id: str
    ticker: str
    as_of: str
    summary: str                                 # 自然语言结论
    claims: list[Claim] = field(default_factory=list)

    def validate(self) -> None:
        if not self.summary.strip():
            raise ContractError("summary 不能为空")
        seen = set()
        for c in self.claims:
            if c.id in seen:
                raise ContractError(f"重复的 claim id: {c.id}")
            seen.add(c.id)
            c.validate()

    def by_metric(self, metric: str) -> Claim | None:
        for c in self.claims:
            if c.metric == metric:
                return c
        return None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Report":
        try:
            claims = [Claim(**c) for c in d.get("claims", [])]
        except TypeError as e:
            raise ContractError(f"claim 字段不符合契约: {e}") from e
        missing = [k for k in ("task_id", "ticker", "as_of", "summary") if k not in d]
        if missing:
            raise ContractError(f"报告缺少字段: {missing}")
        return Report(
            task_id=d["task_id"], ticker=d["ticker"], as_of=d["as_of"],
            summary=d["summary"], claims=claims,
        )

    @staticmethod
    def from_json(s: str) -> "Report":
        try:
            return Report.from_dict(json.loads(s))
        except json.JSONDecodeError as e:
            raise ContractError(f"输出不是合法 JSON: {e}") from e
