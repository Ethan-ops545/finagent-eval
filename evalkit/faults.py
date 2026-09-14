# -*- coding: utf-8 -*-
"""受控故障注入 —— 用来验证「评测器本身」有没有用。

一套评测能给出漂亮的分数，不等于它能发现错误。
所以这里反过来做：拿一份**已知正确**的报告，注入一个**已知种类**的错误，
再看评测器报不报得出来。报不出来，就说明这项检查是摆设。

六种故障各自对应一种真实失效，且刻意都做成"看起来很合理"的样子——
容易看出来的错误没有测试价值。

  wrong_formula     同比算成了倍数（少减 1）。自洽，但与基准复算不符。
  value_typo        公式对、取数对，写报告时数值抄错了。
  unit_swap         数值对，单位标错（ratio 写成 percent）。
  ungrounded_input  引用了一期从未取过的报表，数值照旧（可能碰巧还是对的）。
  lookahead_input   把输入换成 as_of 之后才披露的那一期。
  dropped_metric    该给的结论悄悄不给了。

这不是单元测试的附属品，它就是这个项目想说的那件事：
评测集自己也要被检验。
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable

from finagent.calc import safe_eval
from finagent.schema import Report, Claim
from finagent.snapshot import Snapshots


@dataclass
class Fault:
    key: str
    label: str
    expect_code: str
    apply: Callable[[Report, dict, Snapshots], Report]


def _find(report: Report, metric: str) -> Claim | None:
    return report.by_metric(metric)


def _wrong_formula(report: Report, task: dict, snaps: Snapshots) -> Report:
    r = copy.deepcopy(report)
    c = _find(r, "revenue_growth_yoy")
    if c is None:
        return r
    c.formula = "rev_cur / rev_prev"                     # 少减了 1
    vals = {k: snaps.resolve(v).value for k, v in c.inputs.items()}
    c.value = safe_eval(c.formula, vals)                 # 自洽地错着
    c.text = f"{r.ticker} 收入同比为 {c.value:.4f}"
    return r


def _value_typo(report: Report, task: dict, snaps: Snapshots) -> Report:
    r = copy.deepcopy(report)
    c = _find(r, "net_margin")
    if c is None or c.value is None:
        return r
    c.value = round(c.value * 1.12, 6)                   # 抄错一位
    return r


def _unit_swap(report: Report, task: dict, snaps: Snapshots) -> Report:
    r = copy.deepcopy(report)
    c = _find(r, "gross_margin")
    if c is None:
        return r
    c.unit = "percent"                                   # 数值仍是 0.388，单位却写成百分数
    return r


def _ungrounded_input(report: Report, task: dict, snaps: Snapshots) -> Report:
    r = copy.deepcopy(report)
    c = _find(r, "current_ratio")
    if c is None:
        return r
    for var, ref in list(c.inputs.items()):
        c.inputs[var] = ref.replace("FY2024", "FY2022")  # 这一期 Agent 从没取过
    return r


def _lookahead_input(report: Report, task: dict, snaps: Snapshots) -> Report:
    r = copy.deepcopy(report)
    c = _find(r, "net_margin")
    if c is None:
        return r
    for var, ref in list(c.inputs.items()):
        c.inputs[var] = ref.replace("FY2024", "FY2025")  # as_of 之后才披露
    return r


def _text_contradicts_value(report: Report, task: dict, snaps: Snapshots) -> Report:
    """数值一个字不动，只把给人看的那句话改成方向相反的说法。

    这是最像"没错"的一种错：value 正确、公式正确、输入正确、单位正确、
    都取过、没跨期——前六条检查全绿。但研究员读到的是相反的结论。
    """
    r = copy.deepcopy(report)
    c = _find(r, "revenue_growth_yoy")
    if c is None or c.value is None:
        return r
    direction = "下滑" if c.value >= 0 else "增长"
    c.text = (f"{r.ticker} {task['fiscal_year']} 收入同比{direction} "
              f"{abs(c.value) * 100:.2f}%")
    return r


def _dropped_metric(report: Report, task: dict, snaps: Snapshots) -> Report:
    r = copy.deepcopy(report)
    r.claims = [c for c in r.claims if c.metric != "pe_ratio"]
    return r


FAULTS: list[Fault] = [
    Fault("wrong_formula", "同比少减 1，自洽地算错", "value_mismatch", _wrong_formula),
    Fault("value_typo", "公式对但数值抄错", "self_inconsistent", _value_typo),
    Fault("unit_swap", "数值对但单位标错", "unit_mismatch", _unit_swap),
    Fault("ungrounded_input", "引用了从未取过的一期", "ungrounded", _ungrounded_input),
    Fault("lookahead_input", "用了 as_of 之后披露的数据", "lookahead", _lookahead_input),
    Fault("dropped_metric", "该给的结论没给", "missing_metric", _dropped_metric),
    Fault("text_contradicts_value", "数值全对，但叙述说反了",
          "text_value_mismatch", _text_contradicts_value),
]


# --------------------------------------------------------------------------
# 判断项的故障注入
#
# 上面六种打的是可复算的部分。但报告里还有一半是**不可复算**的表述，
# 那部分由 rubric 打分。rubric 如果从不被检验，它就只是几句好听的话。
#
# 下面两种故障专门破坏表述，而**一个数字都不改**——
# 确定性检查在它们面前会全绿，因为确实没有任何一个数算错。
# 这正是要说明的事：两层评测各管一段，谁也替代不了谁。
# --------------------------------------------------------------------------

@dataclass
class RubricFault:
    key: str
    label: str
    expect_fail_contains: str      # 期望失败的那条标准里应含的关键词
    apply: Callable[[Report, dict], Report]


def _unqualified_verdict(report: Report, task: dict) -> Report:
    """下了跨行业不成立的定论，且不交代依赖什么。数字全对。"""
    r = copy.deepcopy(report)
    r.summary = (f"{r.ticker} {task['fiscal_year']} 收入与盈利表现良好。"
                 f"利息保障倍数与负债率显示公司偿债能力充足，财务安全，无偿债压力。")
    return r


def _one_sided_solvency(report: Report, task: dict) -> Report:
    """只看盈利端，不看杠杆——单看利息保障倍数很容易把高杠杆公司看成安全的。"""
    r = copy.deepcopy(report)
    r.summary = (f"{r.ticker} {task['fiscal_year']} 收入同比增长，毛利率与净利率稳定。"
                 f"利息保障倍数处于行业可比区间，盈利对利息的覆盖程度没有问题。")
    return r


RUBRIC_FAULTS: list[RubricFault] = [
    RubricFault("unqualified_verdict", "下了定论却不交代行业语境",
                "行业语境", _unqualified_verdict),
    RubricFault("one_sided_solvency", "只用盈利端判断偿债，漏掉杠杆",
                "利息保障", _one_sided_solvency),
]
