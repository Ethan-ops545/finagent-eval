# -*- coding: utf-8 -*-
"""确定性评测：凡是能复算的，一律不信 Agent 的话。

六类检查，各自对应一种真实的失效模式：

  missing_metric     该给的结论没给（答非所问）
  value_mismatch     数值与按基准公式独立复算的结果不符（选错公式 / 选错输入）
  self_inconsistent  数值与它自己声明的公式算出来的不符（算完了，写报告时写了别的数）
  unit_mismatch      数值对，单位标错（"13.1" 到底是 13.1% 还是 13.1 倍）
  ungrounded         引用了 Agent 从未取过的数据（凭空捏造；数值可能碰巧是对的）
  lookahead          用了 as_of 之后才披露的数据（前视偏差）

注意 ungrounded 这一条只能靠 trace 查——光比对答案永远发现不了，
因为一个没有依据的结论完全可能恰好是对的。这也是为什么追踪不是"日志"，而是评测的一部分。
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, asdict

from finagent.calc import safe_eval, CalcError
from finagent.schema import Report, Claim
from finagent.snapshot import Snapshots, RefError
from finagent.trace import Trace

REL_TOL = 1e-9

CODES = ("missing_metric", "value_mismatch", "self_inconsistent",
         "unit_mismatch", "ungrounded", "lookahead", "text_value_mismatch", "bad_ref")

# --------------------------------------------------------------------------
# 叙述与数值的一致性
#
# 上面那几条验的全是机器读的字段（value / formula / inputs / unit）。
# 但**没有人会去读 JSON 里的 value**——研究员读的是 text 和 summary。
# 所以一条「value 完全正确、叙述说反了」的结论能毫发无损地通过全部检查，
# 而人拿到手里就是错的。
#
# 这也是大模型的经典失效：生成叙述句和生成结构化字段是两条路径，两边不同步很常见。
# --------------------------------------------------------------------------

# 期间记号必须先整个剔掉，否则「FY2024」里的 2024 会被当成一个数值参与比对。
_PERIOD_TOKEN = re.compile(r"FY\d{4}|\d{4}-\d{2}-\d{2}|\d{4}\s*年|[QH][1-4]")
_NUMBER = re.compile(r"(-?\d+(?:\.\d+)?)\s*(%?)")

UP_WORDS = ("增长", "上升", "提升", "扩大", "改善", "增加")
DOWN_WORDS = ("下滑", "下降", "收窄", "恶化", "减少", "缩小")

# 方向词只对**变化量**类指标有意义。
# 「毛利率 38.8%，较上年下滑」并不矛盾——它说的是变化，不是水平；
# 拿方向词去校验一个水平值的正负号，会误报。所以只对指标名含
# growth / change / yoy / qoq 的做方向校验。这是本条检查的已知边界。
_CHANGE_METRIC = re.compile(r"growth|change|yoy|qoq", re.I)


def _text_numbers(text: str) -> list[tuple[float, int, bool]]:
    """抽出叙述里的候选数值：(数值, 小数位数, 是否带百分号)。"""
    stripped = _PERIOD_TOKEN.sub(" ", text)
    out = []
    for m in _NUMBER.finditer(stripped):
        raw, pct = m.group(1), m.group(2)
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        out.append((float(raw), decimals, pct == "%"))
    return out


def _num_matches(n: float, decimals: int, is_pct: bool,
                 value: float, use_abs: bool) -> bool:
    """容差取叙述自身精度的半个末位——写 0.3880 就按 ±0.00005 判，不苛求全精度。"""
    tol = 0.5 * 10 ** (-decimals) + 1e-9
    targets = [value * 100.0] if is_pct else [value, value * 100.0]
    if use_abs:                      # 叙述用方向词表达正负时，数字写的是绝对值
        n, targets = abs(n), [abs(t) for t in targets]
    return any(abs(n - t) <= tol for t in targets)


def _check_text(c: Claim) -> list[Finding]:
    if c.value is None or not c.text:
        return []
    out: list[Finding] = []
    up = [w for w in UP_WORDS if w in c.text]
    down = [w for w in DOWN_WORDS if w in c.text]

    if _CHANGE_METRIC.search(c.metric or ""):
        if up and c.value < 0:
            out.append(Finding(c.id, "text_value_mismatch",
                               f"叙述说「{up[0]}」，但数值为负",
                               expected="负值应配下降类措辞", actual=c.text))
        if down and c.value > 0:
            out.append(Finding(c.id, "text_value_mismatch",
                               f"叙述说「{down[0]}」，但数值为正",
                               expected="正值应配上升类措辞", actual=c.text))

    nums = _text_numbers(c.text)
    if nums:
        # 有方向词时数字写的是绝对值，正负由词承担，不再重复校验符号。
        use_abs = bool(up or down)
        if not any(_num_matches(n, d, p, c.value, use_abs) for n, d, p in nums):
            out.append(Finding(c.id, "text_value_mismatch",
                               "叙述里的数字与结论的数值对不上",
                               expected=c.value, actual=c.text))
    # 叙述里一个数字都没有时不判错——**无法验证不等于错**。
    return out


@dataclass
class Finding:
    claim_id: str
    code: str
    detail: str
    expected: float | str | None = None
    actual: float | str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=REL_TOL, abs_tol=1e-12)


def ground_truth(spec: dict, snaps: Snapshots) -> float:
    """按基准任务里声明的公式与输入，独立算出正确答案。

    这里刻意不看 Agent 的 formula——评测器必须有自己的一条计算路径，
    否则就成了"用模型的答案验证模型的答案"。
    """
    values = {var: snaps.resolve(ref).value for var, ref in spec["inputs"].items()}
    return safe_eval(spec["formula"], values)


def check_report(report: Report, task: dict, snaps: Snapshots,
                 trace: Trace) -> list[Finding]:
    findings: list[Finding] = []
    exposed = trace.exposed()
    as_of = task["as_of"]
    required = {r["metric"]: r for r in task["require"]}

    # ---- 1. 该给的结论给了没有 ----
    for metric in required:
        if report.by_metric(metric) is None:
            findings.append(Finding(
                claim_id=f"<missing:{metric}>", code="missing_metric",
                detail=f"基准要求的结论 {metric} 未出现在报告中"))

    # ---- 2. 逐条结论检查 ----
    for c in report.claims:
        if c.kind == "judgment":
            continue          # 判断项不在确定性评测范围内，交给 judge.py
        findings.extend(_check_claim(c, required, snaps, exposed, as_of))

    return findings


def _check_claim(c: Claim, required: dict, snaps: Snapshots,
                 exposed: set[str], as_of: str) -> list[Finding]:
    out: list[Finding] = []

    # 2a. 每个输入引用必须能解析、必须被真的取过、必须不晚于 as_of
    values: dict[str, float] = {}
    for var, ref in c.inputs.items():
        try:
            r = snaps.resolve(ref)
        except RefError as e:
            out.append(Finding(c.id, "bad_ref", f"输入 {var} 的引用无法解析: {e}",
                               actual=ref))
            continue
        values[var] = r.value
        if ref not in exposed:
            out.append(Finding(c.id, "ungrounded",
                               f"输入 {var} 引用了 Agent 从未取过的数据",
                               actual=ref))
        if r.published > as_of:
            out.append(Finding(c.id, "lookahead",
                               f"输入 {var} 的发布日 {r.published} 晚于信息可见日 {as_of}",
                               expected=f"<= {as_of}", actual=r.published))

    # 2b. 数值必须与它自己声明的公式一致
    if c.kind == "computed" and c.formula and len(values) == len(c.inputs):
        try:
            own = safe_eval(c.formula, values)
        except CalcError as e:
            out.append(Finding(c.id, "bad_ref", f"公式无法求值: {e}", actual=c.formula))
        else:
            if c.value is not None and not _close(own, c.value):
                out.append(Finding(c.id, "self_inconsistent",
                                   "报告里的数值与它自己声明的公式算出来的不符",
                                   expected=own, actual=c.value))

    # 2c. 与基准独立复算的结果比对（只对基准要求的指标）
    spec = required.get(c.metric or "")
    if spec is not None:
        try:
            truth = ground_truth(spec, snaps)
        except (RefError, CalcError) as e:      # 基准自身写错了，属于评测集的 bug
            out.append(Finding(c.id, "bad_ref", f"基准公式无法求值: {e}"))
        else:
            if c.value is not None and not _close(truth, c.value):
                out.append(Finding(c.id, "value_mismatch",
                                   f"{c.metric} 与基准独立复算结果不符",
                                   expected=truth, actual=c.value))
        if spec.get("unit") and c.unit != spec["unit"]:
            out.append(Finding(c.id, "unit_mismatch",
                               f"{c.metric} 的单位标注不符",
                               expected=spec["unit"], actual=c.unit))

    # 2d. 给人读的那句话，得跟给机器读的那个数一致
    out.extend(_check_text(c))

    return out


def summarize(findings: list[Finding]) -> dict[str, int]:
    counts = {code: 0 for code in CODES}
    for f in findings:
        counts[f.code] = counts.get(f.code, 0) + 1
    return counts
