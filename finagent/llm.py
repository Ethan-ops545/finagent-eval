# -*- coding: utf-8 -*-
"""模型接入层。

两个 provider：

  MockProvider     —— 脚本化的假模型，**不是大模型**。它按固定策略调工具、
                      按固定配方出报告。作用是让整套评测在没有 API key 的环境下
                      也能完整跑通、结果逐次一致。评测器本身要能被测试，
                      就必须有一个行为完全可预测的被测对象。
  AnthropicProvider—— 真接大模型（需要 anthropic 包与 ANTHROPIC_API_KEY）。

两者返回同一种归一化结构，agent.py 不关心背后是谁。
故意没有引入 LangChain 之类的框架，理由见 README「设计取舍」。
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field


@dataclass
class LLMResponse:
    text: str | None = None
    tool_calls: list[dict] = field(default_factory=list)   # [{id, name, input}]

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


# ---------------------------------------------------------------- Mock

# 指标配方。真模型靠财务知识现推，mock 靠这张表。
# 配方写错会被评测器抓出来——faults.py 里的 arith_error 模拟的就是这种情况。
METRIC_RECIPES: dict[str, tuple[str, dict[str, str], str]] = {
    "revenue_growth_yoy": (
        "rev_cur / rev_prev - 1",
        {"rev_cur": "{T}.income.{FY}.revenue", "rev_prev": "{T}.income.{PFY}.revenue"},
        "ratio",
    ),
    "gross_margin": (
        "gp / rev",
        {"gp": "{T}.income.{FY}.gross_profit", "rev": "{T}.income.{FY}.revenue"},
        "ratio",
    ),
    "net_margin": (
        "ni / rev",
        {"ni": "{T}.income.{FY}.net_income", "rev": "{T}.income.{FY}.revenue"},
        "ratio",
    ),
    "current_ratio": (
        "ca / cl",
        {"ca": "{T}.balance.{FY}.current_assets", "cl": "{T}.balance.{FY}.current_liabilities"},
        "ratio",
    ),
    "pe_ratio": (
        "price * shares / ni",
        {"price": "{T}.market.{D}.close_price", "shares": "{T}.market.{D}.shares_outstanding",
         "ni": "{T}.income.{FY}.net_income"},
        "multiple",
    ),
    # 偿债能力。两个指标都只用 FY 当期数据，不跨期。
    # 注意：比率本身可复算，"多少倍算安全"依赖行业结构与债务期限，
    # 不可复算，因此不写成 computed 结论，留给 rubric 判断。
    "interest_coverage": (
        "ebit / interest",
        {"ebit": "{T}.income.{FY}.operating_income",
         "interest": "{T}.income.{FY}.interest_expense"},
        "multiple",
    ),
    "debt_to_assets": (
        "tl / ta",
        {"tl": "{T}.balance.{FY}.total_liabilities",
         "ta": "{T}.balance.{FY}.total_assets"},
        "ratio",
    ),
}


def _fill(tmpl: dict[str, str], spec: dict) -> dict[str, str]:
    fy = spec["fiscal_year"]
    pfy = f"FY{int(fy[2:]) - 1}"
    return {k: v.format(T=spec["ticker"], FY=fy, PFY=pfy, D=spec["as_of"])
            for k, v in tmpl.items()}


class MockProvider:
    """脚本化策略。按轮次推进：列期间 → 取利润表 → 取资产负债表与行情 → 计算 → 出报告。"""

    name = "mock"

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        spec = _task_spec(messages)
        turn = sum(1 for m in messages if m["role"] == "assistant")
        fy = spec["fiscal_year"]
        pfy = f"FY{int(fy[2:]) - 1}"
        t = spec["ticker"]

        if turn == 0:
            return LLMResponse(tool_calls=[
                {"id": "c0", "name": "list_periods", "input": {"ticker": t}}])

        if turn == 1:
            return LLMResponse(tool_calls=[
                {"id": "c1", "name": "get_income_statement", "input": {"ticker": t, "period": fy}},
                {"id": "c2", "name": "get_income_statement", "input": {"ticker": t, "period": pfy}},
            ])

        if turn == 2:
            return LLMResponse(tool_calls=[
                {"id": "c3", "name": "get_balance_sheet", "input": {"ticker": t, "period": fy}},
                {"id": "c4", "name": "get_market_data", "input": {"ticker": t, "date": spec["as_of"]}},
            ])

        if turn == 3:
            calls = []
            for i, m in enumerate(spec["required_metrics"]):
                if m not in METRIC_RECIPES:
                    continue
                expr, tmpl, _ = METRIC_RECIPES[m]
                calls.append({"id": f"m{i}", "name": "compute",
                              "input": {"expression": expr, "inputs": _fill(tmpl, spec)}})
            return LLMResponse(tool_calls=calls)

        return LLMResponse(text=_mock_report(spec, _computed_values(messages)))


def _task_spec(messages: list[dict]) -> dict:
    """从第一条用户消息里取出任务说明（agent.py 以 JSON 块的形式放进去）。"""
    for m in messages:
        if m["role"] != "user":
            continue
        content = m["content"]
        text = content if isinstance(content, str) else "".join(
            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
        match = re.search(r"<task>(.*?)</task>", text, re.S)
        if match:
            return json.loads(match.group(1))
    raise ValueError("对话里找不到 <task> 说明")


def _computed_values(messages: list[dict]) -> dict[str, float]:
    """把历史上所有 compute 工具的返回收集成 {表达式: 值}。"""
    out: dict[str, float] = {}
    for m in messages:
        if not isinstance(m.get("content"), list):
            continue
        for b in m["content"]:
            if not (isinstance(b, dict) and b.get("type") == "tool_result"):
                continue
            try:
                payload = json.loads(b["content"]) if isinstance(b["content"], str) else b["content"]
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(payload, dict) and "expression" in payload and "value" in payload:
                out[payload["expression"]] = payload["value"]
    return out


def _mock_report(spec: dict, values: dict[str, float]) -> str:
    t, fy = spec["ticker"], spec["fiscal_year"]
    claims, got = [], {}
    for i, metric in enumerate(spec["required_metrics"]):
        if metric not in METRIC_RECIPES:
            continue
        expr, tmpl, unit = METRIC_RECIPES[metric]
        if expr not in values:
            continue
        v = values[expr]
        got[metric] = v
        # 变化量类指标用"方向词 + 绝对值百分数"表述，水平类直接报数值。
        # 两种写法都要能被「叙述与数值一致」那条检查验过。
        if "growth" in metric or "yoy" in metric:
            text = (f"{t} {fy} 收入同比{'增长' if v >= 0 else '下滑'} "
                    f"{abs(v) * 100:.2f}%")
        else:
            text = f"{t} {fy} 的 {metric} 为 {v:.4f}"
        claims.append({
            "id": f"k{i}", "kind": "computed", "metric": metric,
            "text": text,
            "value": v, "unit": unit, "formula": expr, "inputs": _fill(tmpl, spec),
        })
    claims.append({
        "id": "j0", "kind": "judgment", "metric": None,
        "text": ("增长与盈利能力两条线需要合看：收入增速反映规模扩张，"
                 "毛利率与净利率反映这轮扩张有没有以牺牲单位经济性为代价。"),
        "value": None, "unit": None, "formula": None, "inputs": {},
    })
    # 偿债能力的判断刻意放在 judgment：比率可复算，"多少算安全"不可复算。
    claims.append({
        "id": "j1", "kind": "judgment", "metric": None,
        "text": ("偿债能力不能只看单一比率：负债率衡量杠杆水平，利息保障倍数衡量"
                 "当期盈利对利息的覆盖程度，二者背离时才说明问题。"
                 "阈值本身依赖行业的资产结构与债务期限，没有跨行业通用的安全线。"),
        "value": None, "unit": None, "formula": None, "inputs": {},
    })
    g = got.get("revenue_growth_yoy")
    gm = got.get("gross_margin")
    nm = got.get("net_margin")
    ic = got.get("interest_coverage")
    da = got.get("debt_to_assets")
    summary = (
        f"以下结论基于 {t} 的 {fy} 年报，信息可见日为 {spec['as_of']}，"
        f"不包含该日之后披露的任何数据。"
        f"收入同比{'增长' if (g or 0) >= 0 else '下滑'} {abs(g or 0) * 100:.1f}%；"
        f"毛利率 {(gm or 0) * 100:.1f}%，净利率 {(nm or 0) * 100:.1f}%，"
        f"盈利能力与增长需合并判断。"
    )
    if ic is not None and da is not None:
        summary += (
            f"偿债端，利息保障倍数 {ic:.1f} 倍、负债率 {da * 100:.1f}%，两者须合看——"
            f"负债率反映杠杆水平，利息保障反映当期盈利对利息的覆盖程度；"
            f"是否安全取决于所处行业的资产结构与债务期限，"
            f"本报告不给出跨行业可比的定论。"
        )
    summary += "本报告不对下一财年作任何预测——现有数据不足以支撑前瞻结论。"
    report = {"task_id": spec["task_id"], "ticker": t, "as_of": spec["as_of"],
              "summary": summary, "claims": claims}
    return "```json\n" + json.dumps(report, ensure_ascii=False, indent=2) + "\n```"


# ---------------------------------------------------------------- Anthropic

class AnthropicProvider:
    """真模型。需要 `pip install anthropic` 且设置 ANTHROPIC_API_KEY。"""

    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-4-20250514", max_tokens: int = 4096):
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError("未安装 anthropic 包：pip install anthropic") from e
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("未设置环境变量 ANTHROPIC_API_KEY")
        self._client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        resp = self._client.messages.create(
            model=self.model, max_tokens=self.max_tokens,
            system=system, messages=messages, tools=tools,
        )
        text_parts, calls = [], []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append({"id": block.id, "name": block.name, "input": block.input})
        return LLMResponse(text="\n".join(text_parts) if text_parts else None,
                           tool_calls=calls)


def get_provider(name: str = "mock", **kwargs):
    if name == "mock":
        return MockProvider()
    if name == "anthropic":
        return AnthropicProvider(**kwargs)
    raise ValueError(f"未知 provider: {name}")
