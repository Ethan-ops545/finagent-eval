# -*- coding: utf-8 -*-
"""端到端演示入口。

    python run_demo.py                 # 默认 mock provider，无需任何 API key
    python run_demo.py --provider anthropic --model claude-sonnet-4-20250514

产物写在 runs/ 下：
    runs/scorecard.md          记分卡（先看第一节「评测集自检」）
    runs/scorecard.json        同上，机器可读
    runs/<task>.report.json    Agent 的结构化报告
    runs/<task>.trace.jsonl    该次运行的全链路追踪
"""
from __future__ import annotations

import argparse
import sys

from evalkit.runner import main as run_main


def main() -> int:
    ap = argparse.ArgumentParser(description="金融研究 Agent 评测 Demo")
    ap.add_argument("--provider", default="mock", choices=["mock", "anthropic"],
                    help="mock 为脚本化假模型（默认，可离线复现）")
    args = ap.parse_args()

    res = run_main(args.provider)
    st = res["selftest"]

    # 退出码按"评测器自己是否可信"来定，而不是按 Agent 考了多少分。
    # 两种失败要分开，因为处置方式不同：
    #   1  确定性检查器有问题（误报或漏报）—— 分数不可用，要修 evalkit
    #   2  参考实现与基准脱节 —— 分数可用（它确实少答了），但要同步实现
    #   3  判分器有问题 —— 确定性那半可用，判断项那半不可用，要修 judge/rubric
    if not st["checker_sound"]:
        print(f"\n检查器自检未通过（误报 {st['false_positives']} 条，"
              f"故障检出 {st['detected']}/{st['total']}）：这轮分数不可用，"
              f"要修的是 evalkit 本身。", file=sys.stderr)
        return 1

    if not st["rubric_sound"]:
        print(f"\n确定性检查可信，但判分器自检未通过"
              f"（参照答案失分 {len(st['rubric_false_positives'])} 条，"
              f"缺陷检出 {st['rubric_detected']}/{st['rubric_total']}）："
              f"第三节的分数不可用，要修的是 judge 或 rubric。", file=sys.stderr)
        return 3

    if st["baseline_checked"] and not st["baseline_healthy"]:
        print(f"\n检查器可信（零误报、已知故障全检出），但参考实现与基准脱节："
              f"{st['baseline_findings']} 条 findings。\n"
              f"分数是真实的（它确实没答全），但你多半是加了基准指标却没补 "
              f"METRIC_RECIPES。", file=sys.stderr)
        return 2

    print("\n自检通过：确定性检查零误报、已知故障全检出；"
          f"判分器零误判、已知表述缺陷全检出（{st['rubric_detected']}/{st['rubric_total']}）"
          + ("；参考实现与基准同步" if st["baseline_checked"] else "")
          + "。上面的分数可以采信。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
