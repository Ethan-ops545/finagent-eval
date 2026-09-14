# -*- coding: utf-8 -*-
"""参考报告（oracle）—— 一份「在构造上必然正确」的报告。

为什么需要它：

要测"检查器会不会误报"，就得有一份**确知正确**的报告。
拿被测 Agent 跑出来的报告来测是循环论证——Agent 对不对本身是未知数，
它上面报出的 finding 到底是误报，还是 Agent 真的错了，分不出来。

所以这里不经过 Agent，直接按基准自己声明的 formula 与 inputs 把每个值算出来，
单位照抄基准，输入地址照抄基准，再配一份恰好暴露了这些地址的追踪。
这样组装出来的报告，按定义不可能违反任何一条检查。
**检查器在它上面报出的任何 finding，都是货真价实的误报。**

同理，故障注入也打在这份报告上，而不是打在 mock 的报告上——
"能不能检出已知故障"这件事，不该依赖被测对象的好坏。
"""
from __future__ import annotations

import time

from finagent.schema import Report, Claim
from finagent.snapshot import Snapshots
from finagent.trace import Trace, ToolCall
from .deterministic import ground_truth


def build_oracle(task: dict, snaps: Snapshots) -> tuple[Report, Trace]:
    """按基准自身的声明合成参考报告，以及一份与之匹配的追踪。"""
    claims: list[Claim] = []
    refs: list[str] = []

    for i, spec in enumerate(task["require"]):
        truth = ground_truth(spec, snaps)
        claims.append(Claim(
            id=f"o{i}",
            kind="computed",
            metric=spec["metric"],
            # 叙述里必须带上数值：参考报告要在**每一条**检查上都成立，
            # 包括「叙述与数值一致」这条。只写指标名会让那条检查无从验起。
            text=(f"[oracle] {task['ticker']} {task['fiscal_year']} 的 "
                  f"{spec['metric']} 为 {truth:.4f}"),
            value=truth,
            unit=spec.get("unit"),
            formula=spec["formula"],
            inputs=dict(spec["inputs"]),
        ))
        refs.extend(spec["inputs"].values())

    report = Report(
        task_id=task["id"], ticker=task["ticker"], as_of=task["as_of"],
        summary=(f"[oracle] 按基准声明直接复算的参考报告，"
                 f"标的 {task['ticker']}，期间 {task['fiscal_year']}，"
                 f"信息可见日 {task['as_of']}。"),
        claims=claims,
    )
    report.validate()

    # 一份"恰好取到了基准所需数据、没多取也没少取"的追踪。
    # 多取会让 ungrounded 检查失去意义，少取会造成误报——两头都不能有。
    trace = Trace(task_id=task["id"], provider="oracle")
    trace.add(ToolCall(step=1, tool="<oracle>", args={"note": "按基准声明直接取数"},
                       ok=True, latency_ms=0.0, exposed_refs=sorted(set(refs))))
    trace.finished_at = time.time()
    return report, trace
