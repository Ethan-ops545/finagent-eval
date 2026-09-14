# -*- coding: utf-8 -*-
"""评测器的测试。被测对象不是 Agent，是**评测器自己**。

四组主张，对应四件不同的事：

  A.  检查器可信      面对构造上必然正确的参考报告，零 finding（零误报）；
                      面对六种已知故障，每种都抓得到（零漏报）。
  B.  参考实现同步    mock 是本基准的参考实现，它应当拿满分。
  B2. 判分器可信      参照答案零失分；注入已知表述缺陷必须失分；
                      且这些缺陷对确定性检查**不可见**——两层互补，不是冗余。
  C.  两者可分        基准与实现脱节时，必须表现为 B 失败而 A 仍然通过——
                      这两种失败的修法不同，混在一起就分不出该修哪边。

C 是回归测试：早先的版本把 A 和 B 合成了一个"零误报"计数器，
结果基准一加指标就误判成"检查器有问题"。

跑法: python -m pytest tests/ -v     或直接 python tests/test_harness.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

from finagent.agent import run_task
from finagent.calc import safe_eval, CalcError
from finagent.llm import get_provider
from finagent.snapshot import Snapshots
from evalkit import deterministic as det
from evalkit.faults import FAULTS, RUBRIC_FAULTS
from evalkit.judge import HeuristicJudge
from evalkit.oracle import build_oracle
from evalkit.runner import load_tasks, run_selftest


@pytest.fixture(scope="module")
def snaps():
    return Snapshots()


@pytest.fixture(scope="module")
def task():
    return load_tasks()[0]


@pytest.fixture(scope="module")
def oracle_run(task, snaps):
    """构造上必然正确的参考报告。测"误报"只能拿它测。"""
    report, trace = build_oracle(task, snaps)
    return task, report, trace, snaps


@pytest.fixture(scope="module")
def baseline_run(task, snaps):
    """参考实现（mock）实际跑出来的报告。"""
    report, trace = run_task(task, snaps, get_provider("mock"))
    return task, report, trace, snaps


# ------------------------------------------------- A. 检查器可信（零误报）

def test_oracle_report_produces_no_false_positives(oracle_run):
    task, report, trace, snaps = oracle_run
    findings = det.check_report(report, task, snaps, trace)
    assert findings == [], f"检查器在必然正确的报告上误报: {[f.to_dict() for f in findings]}"


def test_oracle_covers_every_required_metric(oracle_run):
    """参考报告漏了指标的话，上面那条测试会因为覆盖不全而虚假通过。"""
    task, report, _, _ = oracle_run
    for spec in task["require"]:
        assert report.by_metric(spec["metric"]) is not None


# ------------------------------------------------- A. 检查器可信（零漏报）

@pytest.mark.parametrize("fault", FAULTS, ids=[f.key for f in FAULTS])
def test_each_injected_fault_is_detected(fault, oracle_run):
    task, report, trace, snaps = oracle_run
    mutated = fault.apply(report, task, snaps)
    codes = {f.code for f in det.check_report(mutated, task, snaps, trace)}
    assert fault.expect_code in codes, (
        f"注入 {fault.key} 后未检出 {fault.expect_code}，只报出 {sorted(codes)}")


def test_faults_actually_change_the_report(oracle_run):
    """反向保险：某个注入器若其实什么都没改，上面的测试就是在自欺欺人。"""
    task, report, _, snaps = oracle_run
    for fault in FAULTS:
        mutated = fault.apply(report, task, snaps)
        assert mutated.to_json() != report.to_json(), f"{fault.key} 没有改动报告"


# ------------------------------------------------- A. 叙述一致性的边界

def _claim(**kw):
    from finagent.schema import Claim
    base = dict(id="t", kind="computed", metric="gross_margin", text="",
                value=0.388, unit="ratio", formula="gp / rev",
                inputs={"gp": "X.income.FY2024.gross_profit",
                        "rev": "X.income.FY2024.revenue"})
    base.update(kw)
    return Claim(**base)


def test_period_tokens_in_text_are_not_read_as_values():
    """「FY2024」里的 2024 不能被当成一个数值——否则每条干净结论都会误报。"""
    from evalkit.deterministic import _check_text
    c = _claim(text="NOVA FY2024 的 gross_margin 为 0.3880", value=0.38799999)
    assert _check_text(c) == []


def test_text_number_that_really_disagrees_is_caught():
    from evalkit.deterministic import _check_text
    c = _claim(text="NOVA FY2024 的 gross_margin 为 0.5100", value=0.38799999)
    assert [f.code for f in _check_text(c)] == ["text_value_mismatch"]


def test_percent_rendering_is_accepted():
    """0.388 写成「38.80%」是对的，不能因为字面不等就判错。"""
    from evalkit.deterministic import _check_text
    assert _check_text(_claim(text="毛利率 38.80%", value=0.38799999)) == []


def test_direction_word_must_agree_with_sign_on_change_metrics():
    from evalkit.deterministic import _check_text
    c = _claim(metric="revenue_growth_yoy", text="收入同比下滑 31.00%", value=0.31)
    assert any(f.code == "text_value_mismatch" for f in _check_text(c))


def test_direction_check_does_not_fire_on_level_metrics():
    """「毛利率 38.80%，较上年下滑」说的是变化不是水平，不构成矛盾。

    这是本条检查的已知边界：方向词只对变化量类指标校验。
    """
    from evalkit.deterministic import _check_text
    c = _claim(metric="gross_margin", text="毛利率 38.80%，较上年下滑", value=0.38799999)
    assert _check_text(c) == []


def test_text_without_numbers_is_not_penalised():
    """无法验证不等于错。"""
    from evalkit.deterministic import _check_text
    assert _check_text(_claim(text="毛利率处于同业中游水平")) == []


# ------------------------------------------------- B. 参考实现与基准同步

def test_baseline_output_satisfies_contract(baseline_run):
    _, report, _, _ = baseline_run
    report.validate()
    assert any(c.kind == "computed" for c in report.claims)


def test_baseline_claims_are_all_grounded(baseline_run):
    _, report, trace, _ = baseline_run
    exposed = trace.exposed()
    for c in report.claims:
        for ref in c.inputs.values():
            assert ref in exposed, f"{c.id} 引用了未取过的数据: {ref}"


def test_baseline_is_in_sync_with_benchmark(baseline_run):
    """mock 是参考实现，应当拿满分。不满分通常意味着基准加了指标而配方没补。"""
    task, report, trace, snaps = baseline_run
    findings = det.check_report(report, task, snaps, trace)
    assert findings == [], (
        "参考实现与基准脱节（不是检查器误报）: "
        f"{[f.to_dict() for f in findings]}")


# ------------------------------------------------- B2. 判断项（rubric）也要被检验

def test_reference_answer_passes_every_rubric_criterion(baseline_run):
    task, report, _, _ = baseline_run
    scores = HeuristicJudge().score(report, task)
    assert scores, "rubric 为空，下面的检验没有意义"
    failed = [s.criterion for s in scores if not s.passed]
    assert failed == [], f"判分器把参照答案判失分了（误判）: {failed}"


@pytest.mark.parametrize("rf", RUBRIC_FAULTS, ids=[f.key for f in RUBRIC_FAULTS])
def test_each_rubric_fault_is_detected(rf, baseline_run):
    task, report, _, _ = baseline_run
    mutated = rf.apply(report, task)
    failed = [s.criterion for s in HeuristicJudge().score(mutated, task) if not s.passed]
    assert any(rf.expect_fail_contains in c for c in failed), (
        f"注入 {rf.key} 后，没有任何含「{rf.expect_fail_contains}」的标准失分；"
        f"实际失分的是 {failed}")


@pytest.mark.parametrize("rf", RUBRIC_FAULTS, ids=[f.key for f in RUBRIC_FAULTS])
def test_rubric_faults_are_invisible_to_deterministic_checks(rf, baseline_run):
    """两层评测互补，不是冗余。

    这两种故障只改表述、一个数字都不改，所以确定性检查必须全绿。
    要是确定性检查也报了问题，说明故障注入写脏了，rubric 那边的检出就不算数。
    """
    task, report, trace, snaps = baseline_run
    mutated = rf.apply(report, task)
    findings = det.check_report(mutated, task, snaps, trace)
    assert findings == [], (
        f"{rf.key} 本该只破坏表述，却触发了确定性检查: "
        f"{[f.to_dict() for f in findings]}")


# ------------------------------------------------- C. 两种失败必须可分辨

def test_selftest_separates_checker_soundness_from_baseline_sync(monkeypatch):
    """回归测试：人为制造"基准有、实现没有"的脱节，
    自检必须报成「参考实现脱节」，而不是「检查器误报」。"""
    import finagent.llm as llm

    metric = "gross_margin"
    assert metric in llm.METRIC_RECIPES
    monkeypatch.delitem(llm.METRIC_RECIPES, metric)

    st = run_selftest("mock")

    assert st["checker_sound"] is True, "实现脱节不该被算成检查器的问题"
    assert st["false_positives"] == 0, "参考报告不受实现脱节影响，不该出现误报"
    assert st["detected"] == st["total"], "故障检出不该受实现脱节影响"
    assert st["baseline_healthy"] is False, "实现确实脱节了，本项应当失败"
    assert any(r["code"] == "missing_metric" for r in st["baseline_rows"])


def test_selftest_passes_on_clean_repo():
    st = run_selftest("mock")
    assert st["checker_sound"] is True
    assert st["false_positives"] == 0
    assert st["detected"] == st["total"]
    assert st["baseline_healthy"] is True
    assert st["rubric_sound"] is True
    assert st["rubric_false_positives"] == []
    assert st["rubric_detected"] == st["rubric_total"] > 0


# ------------------------------------------------- 求值器安全性

def test_calculator_rejects_code_execution():
    """公式字符串来自大模型，属于不可信输入。"""
    for bad in ["__import__('os').system('echo hi')", "open('/etc/passwd').read()",
                "(1).__class__", "[x for x in range(3)]", "a if a else 0"]:
        with pytest.raises(CalcError):
            safe_eval(bad, {"a": 1.0})


def test_calculator_basic_math():
    assert safe_eval("a / b - 1", {"a": 110.0, "b": 100.0}) == pytest.approx(0.1)
    with pytest.raises(CalcError):
        safe_eval("a / b", {"a": 1.0, "b": 0.0})
    with pytest.raises(CalcError):
        safe_eval("a + missing", {"a": 1.0})


# ------------------------------------------------- 数据自洽

def test_snapshot_accounting_identity_holds(snaps):
    """资产 = 负债 + 所有者权益。基准数据自己先得是对的。"""
    for t in snaps.tickers():
        for period, b in snaps.load(t)["statements"]["balance"].items():
            diff = b["total_assets"] - b["total_liabilities"] - b["total_equity"]
            assert abs(diff) < 1e-6, f"{t} {period} 资产负债表不平: {diff}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
