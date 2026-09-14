# -*- coding: utf-8 -*-
"""跑基准、出记分卡。

记分卡刻意分成两半，**不合并成一个总分**：

  确定性部分  能复算的结论，对了多少条；以及接地性、单位、前视三项各有多少违规。
  判断部分    rubric 通过率，单独列。

再加一节「评测集自检」：注入六种已知故障，看评测器抓到几种。
这一节是给读记分卡的人看的——分数高不高先放一边，
先回答"这套评测到底能不能发现错误"。
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field

import yaml

from finagent.agent import run_task
from finagent.llm import get_provider
from finagent.snapshot import Snapshots
from . import deterministic as det
from .faults import FAULTS, RUBRIC_FAULTS
from .judge import HeuristicJudge
from .oracle import build_oracle

ROOT = pathlib.Path(__file__).resolve().parents[1]
TASKS = ROOT / "benchmark" / "tasks.yaml"
RUNS = ROOT / "runs"


def load_tasks(path: pathlib.Path | str = TASKS) -> list[dict]:
    return yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8"))


@dataclass
class TaskResult:
    task_id: str
    ticker: str
    required: int
    correct: int
    findings: list[dict] = field(default_factory=list)
    counts: dict = field(default_factory=dict)
    rubric: list[dict] = field(default_factory=list)
    tool_calls: int = 0
    tool_errors: int = 0
    llm_turns: int = 0
    error: str | None = None


def run_benchmark(provider_name: str = "mock", judge=None,
                  tasks_path: pathlib.Path | str = TASKS) -> dict:
    snaps = Snapshots()
    provider = get_provider(provider_name)
    judge = judge or HeuristicJudge()
    tasks = load_tasks(tasks_path)
    results: list[TaskResult] = []
    RUNS.mkdir(exist_ok=True)

    for task in tasks:
        try:
            report, trace = run_task(task, snaps, provider)
        except Exception as e:                      # 契约违规也是一种失败，要记下来
            results.append(TaskResult(task["id"], task["ticker"],
                                      len(task["require"]), 0,
                                      error=f"{type(e).__name__}: {e}"))
            continue
        trace.write_jsonl(RUNS / f"{task['id']}.trace.jsonl")
        findings = det.check_report(report, task, snaps, trace)
        bad = {f.claim_id for f in findings}
        correct = sum(1 for r in task["require"]
                      if (c := report.by_metric(r["metric"])) is not None
                      and c.id not in bad)
        results.append(TaskResult(
            task_id=task["id"], ticker=task["ticker"],
            required=len(task["require"]), correct=correct,
            findings=[f.to_dict() for f in findings],
            counts=det.summarize(findings),
            rubric=[s.to_dict() for s in judge.score(report, task)],
            tool_calls=len(trace.calls), tool_errors=trace.tool_error_count(),
            llm_turns=trace.llm_turns,
        ))
        (RUNS / f"{task['id']}.report.json").write_text(
            report.to_json(), encoding="utf-8")

    selftest = run_selftest(provider_name, tasks_path)
    return {"provider": provider_name,
            "tasks": [r.__dict__ for r in results],
            "selftest": selftest}


def run_selftest(provider_name: str = "mock",
                 tasks_path: pathlib.Path | str = TASKS) -> dict:
    """自检，两件独立的事，**不可混为一谈**：

    一、检查器是否可信
        在参考报告（构造上必然正确）上必须零 finding —— 有就是误报；
        往参考报告里注入六种已知故障，每种都必须被检出 —— 漏一种就是漏报。
        这一组只考检查器，与被测 Agent 无关。

    二、参考实现是否与基准同步
        mock 是这个基准的参考实现，它应当拿满分。拿不满分说明基准与实现
        脱节（例如基准新增了指标而 METRIC_RECIPES 没跟上），而**不是**检查器误报。
        接了真模型时这条不适用——真模型拿不满分是正常的测量结果，不是配置错误。

    早先的版本把这两件事合并成一个"零误报"计数器，结果基准一加指标就报
    "评测集自检未通过"，而真实原因是参考实现没跟上。两种失败的处置方式完全不同：
    前者要修检查器，后者要同步实现。混在一起就分不出该修哪边。
    """
    snaps = Snapshots()
    task = load_tasks(tasks_path)[0]

    # ---- 一、检查器可信度（只用参考报告，不经过 Agent）----
    oracle, oracle_trace = build_oracle(task, snaps)
    fp = det.check_report(oracle, task, snaps, oracle_trace)

    rows = []
    for f in FAULTS:
        mutated = f.apply(oracle, task, snaps)
        codes = {x.code for x in det.check_report(mutated, task, snaps, oracle_trace)}
        rows.append({"fault": f.key, "label": f.label,
                     "expect": f.expect_code, "detected": f.expect_code in codes,
                     "codes": sorted(codes)})
    detected = sum(1 for r in rows if r["detected"])
    checker_sound = (len(fp) == 0) and (detected == len(rows))

    # ---- 二、参考实现同步性 ----
    # mock 总是跑一遍（代价为零），它同时充当第三节 rubric 自检的参照答案。
    # 但"同步与否"这个判断只在被测就是 mock 时才有意义。
    is_reference = provider_name == "mock"
    baseline: list = []
    baseline_error = None
    ref_report = None
    try:
        ref_report, ref_trace = run_task(task, snaps, get_provider("mock"))
        baseline = det.check_report(ref_report, task, snaps, ref_trace)
    except Exception as e:
        baseline_error = f"{type(e).__name__}: {e}"

    # ---- 三、判断项是否有效 ----
    # rubric 不被检验的话就只是几句好听的话。这里同样用"参照答案零失分 +
    # 注入已知表述缺陷必须失分"来验它。注意这两个故障**一个数字都不改**，
    # 所以确定性检查在它们面前全绿——两层评测各管一段，谁也替代不了谁。
    judge = HeuristicJudge()
    rubric_fp: list = []
    rubric_rows: list = []
    if ref_report is not None:
        rubric_fp = [s.to_dict() for s in judge.score(ref_report, task) if not s.passed]
        for rf in RUBRIC_FAULTS:
            mutated = rf.apply(ref_report, task)
            failed = [s.criterion for s in judge.score(mutated, task) if not s.passed]
            det_codes = {f.code for f in det.check_report(mutated, task, snaps, ref_trace)}
            rubric_rows.append({
                "fault": rf.key, "label": rf.label,
                "expect": rf.expect_fail_contains,
                "detected": any(rf.expect_fail_contains in c for c in failed),
                "failed": failed,
                "deterministic_codes": sorted(det_codes),
            })
    rubric_detected = sum(1 for r in rubric_rows if r["detected"])
    rubric_sound = (not rubric_fp) and rubric_rows and rubric_detected == len(rubric_rows)

    return {
        "false_positives": len(fp),
        "false_positive_rows": [x.to_dict() for x in fp],
        "detected": detected,
        "total": len(rows),
        "rows": rows,
        "checker_sound": checker_sound,
        "baseline_checked": is_reference,
        "baseline_findings": len(baseline),
        "baseline_rows": [x.to_dict() for x in baseline],
        "baseline_error": baseline_error,
        "baseline_healthy": is_reference and not baseline_error and len(baseline) == 0,
        "rubric_false_positives": rubric_fp,
        "rubric_rows": rubric_rows,
        "rubric_detected": rubric_detected,
        "rubric_total": len(rubric_rows),
        "rubric_sound": bool(rubric_sound),
    }


def to_markdown(res: dict) -> str:
    L = ["# 评测记分卡", "", f"被测 provider：`{res['provider']}`", ""]

    st = res["selftest"]
    L += ["## 一、自检（先看这个）", "",
          "### 1. 检查器是否可信", "",
          "在**参考报告**上考：这份报告直接按基准声明的公式复算而来，构造上必然正确，",
          "不经过被测 Agent。所以它上面的任何 finding 都是误报。", "",
          f"- 误报：**{st['false_positives']}**"
          f"（{'无' if st['false_positives'] == 0 else '⚠️ 检查器有 bug'}）",
          f"- 已知故障检出：**{st['detected']} / {st['total']}**", "",
          "| 注入的故障 | 说明 | 期望检出 | 结果 | 实际报出的问题 |",
          "|---|---|---|---|---|"]
    for r in st["rows"]:
        L.append(f"| `{r['fault']}` | {r['label']} | `{r['expect']}` | "
                 f"{'✅' if r['detected'] else '❌'} | {', '.join(r['codes']) or '—'} |")
    if st["false_positive_rows"]:
        L += ["", "误报明细：", "", "| 结论 | 类型 | 说明 |", "|---|---|---|"]
        for f in st["false_positive_rows"]:
            L.append(f"| `{f['claim_id']}` | `{f['code']}` | {f['detail']} |")
    L += ["", "> 有误报或有漏报时，下面的分数一律不看——"
              "一套抓不到已知错误的评测，给出的高分只说明它没在看；"
              "一套会误报的评测，报出来的问题也不值得查。", ""]

    L += ["### 2. 参考实现是否与基准同步", ""]
    if not st["baseline_checked"]:
        L += ["被测不是参考实现（mock），本项不适用——真模型拿不满分是正常的测量结果，"
              "不是配置错误。", ""]
    elif st["baseline_error"]:
        L += [f"- ⚠️ 参考实现运行失败：{st['baseline_error']}", ""]
    else:
        L += [f"- 参考实现在基准上的 findings：**{st['baseline_findings']}**"
              f"（{'同步' if st['baseline_healthy'] else '⚠️ 与基准脱节'}）", ""]
        if st["baseline_rows"]:
            L += ["这**不是**检查器误报，是参考实现没跟上基准（例如基准新增了指标，"
                  "而 `METRIC_RECIPES` 没补配方）。要修的是实现，不是检查器。", "",
                  "| 结论 | 类型 | 说明 |", "|---|---|---|"]
            for f in st["baseline_rows"]:
                L.append(f"| `{f['claim_id']}` | `{f['code']}` | {f['detail']} |")
            L.append("")

    L += ["### 3. 判断项是否有效", "",
          f"- 参照答案的 rubric 失分：**{len(st['rubric_false_positives'])}**"
          f"（{'无' if not st['rubric_false_positives'] else '⚠️ 判分器把对的判成错的'}）",
          f"- 已知表述缺陷检出：**{st['rubric_detected']} / {st['rubric_total']}**", "",
          "| 注入的缺陷 | 说明 | 期望失分的标准 | 结果 | 确定性检查报了什么 |",
          "|---|---|---|---|---|"]
    for r in st["rubric_rows"]:
        codes = ", ".join(r["deterministic_codes"]) or "**全绿**"
        L.append(f"| `{r['fault']}` | {r['label']} | 含「{r['expect']}」 | "
                 f"{'✅' if r['detected'] else '❌'} | {codes} |")
    L += ["", "> 最后一列是重点：这两个故障**一个数字都没改**，所以确定性检查全绿。"
              "光靠复算抓不到「把需要行业语境的判断说成确定结论」这类错误——"
              "而在投研场景里，这种错误比算错一个小数点贵得多。", ""]

    L += ["## 二、确定性评测（可复算部分）", "",
          "| 任务 | 标的 | 正确 / 应答 | 接地性 | 单位 | 前视 | 自洽 | 工具调用 | 模型轮次 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for t in res["tasks"]:
        if t["error"]:
            L.append(f"| {t['task_id']} | {t['ticker']} | 运行失败 | — | — | — | — | — | — |")
            continue
        c = t["counts"]
        L.append(f"| {t['task_id']} | {t['ticker']} | {t['correct']} / {t['required']} | "
                 f"{c.get('ungrounded', 0)} | {c.get('unit_mismatch', 0)} | "
                 f"{c.get('lookahead', 0)} | {c.get('self_inconsistent', 0)} | "
                 f"{t['tool_calls']}（错 {t['tool_errors']}） | {t['llm_turns']} |")

    L += ["", "## 三、判断项（rubric，单独计分）", "",
          "| 任务 | 通过 / 全部 | 未通过的标准 |", "|---|---|---|"]
    for t in res["tasks"]:
        if t["error"]:
            continue
        passed = sum(1 for s in t["rubric"] if s["passed"])
        failed = [s["criterion"] for s in t["rubric"] if not s["passed"]]
        L.append(f"| {t['task_id']} | {passed} / {len(t['rubric'])} | "
                 f"{'；'.join(failed) if failed else '—'} |")

    L += ["", "> 判断项**不并入**确定性得分。两者合成一个总分，"
              "等于允许一段写得好的结论把一个算错的数字盖过去。", ""]

    fails = [(t["task_id"], f) for t in res["tasks"] for f in t.get("findings", [])]
    if fails:
        L += ["## 四、问题明细", "", "| 任务 | 结论 | 类型 | 说明 | 期望 | 实际 |",
              "|---|---|---|---|---|---|"]
        for tid, f in fails:
            L.append(f"| {tid} | `{f['claim_id']}` | `{f['code']}` | {f['detail']} | "
                     f"{f['expected']} | {f['actual']} |")
        L.append("")
    return "\n".join(L)


def main(provider_name: str = "mock") -> dict:
    res = run_benchmark(provider_name)
    RUNS.mkdir(exist_ok=True)
    (RUNS / "scorecard.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    md = to_markdown(res)
    (RUNS / "scorecard.md").write_text(md, encoding="utf-8")
    print(md)
    return res
