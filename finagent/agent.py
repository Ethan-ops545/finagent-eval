# -*- coding: utf-8 -*-
"""Agent 主循环。

手写的工具调用循环，没有用任何 Agent 框架（理由见 README）。
一轮的流程是：模型回一批 tool_use → 逐个执行并记 trace → 把 tool_result 回填进对话 →
再问模型。直到模型给出最终文本，或者达到轮次上限。

轮次上限不是可选项：Agent 的失败模式之一就是在工具之间反复兜圈子，
没有硬上限的循环在生产里等于一个没有超时的 while True。
"""
from __future__ import annotations

import json
import re
import time

from .llm import LLMResponse
from .schema import Report, ContractError
from .snapshot import Snapshots
from .tools import Toolbox, TOOL_SPECS
from .trace import Trace

SYSTEM = """你是一个金融研究助理 Agent。你的输出会被自动评测，所以必须严格守下面的规则。

【信息可见日】任务会给出 as_of。你只能使用发布日期不晚于 as_of 的数据。
取数工具不会替你拦截晚于 as_of 的数据——它返回的 published 字段就是发布日，请你自己判断。
使用了 as_of 之后才披露的数据，属于前视偏差，会被判为错误。

【不准心算】任何算术都必须调用 compute 工具。你要提交公式和每个变量对应的字段引用
（格式 TICKER.statement.PERIOD.field），由工具回到原始数据取值计算。
报告里写的数值必须等于 compute 的返回值。

【结论必须分类】最终输出一个 JSON 报告，每条结论都要标明类别：
  computed  —— 算出来的。必须带 formula 和 inputs。
  retrieved —— 直接取的。必须带恰好一个 input。
  judgment  —— 判断性表述。不带数值。
凡是没有取数工具支撑过的字段引用，都不许出现在 inputs 里。

【输出格式】最后一轮只输出一个 ```json 代码块，结构如下：
{
  "task_id": "...", "ticker": "...", "as_of": "YYYY-MM-DD",
  "summary": "自然语言结论",
  "claims": [
    {"id":"k0","kind":"computed","metric":"gross_margin","text":"...",
     "value":0.3881,"unit":"ratio","formula":"gp / rev",
     "inputs":{"gp":"NOVA.income.FY2024.gross_profit","rev":"NOVA.income.FY2024.revenue"}}
  ]
}
"""


def build_user_message(task: dict) -> str:
    spec = {
        "task_id": task["id"],
        "ticker": task["ticker"],
        "as_of": task["as_of"],
        "fiscal_year": task["fiscal_year"],
        "question": task["question"],
        "required_metrics": [r["metric"] for r in task["require"]],
    }
    return (
        f"{task['question']}\n\n"
        f"请给出下列指标，并按契约输出 JSON 报告。\n"
        f"<task>{json.dumps(spec, ensure_ascii=False)}</task>"
    )


def _extract_json(text: str) -> str:
    m = re.search(r"```json\s*(.*?)\s*```", text, re.S)
    if m:
        return m.group(1)
    m = re.search(r"```\s*(\{.*?\})\s*```", text, re.S)
    if m:
        return m.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return text[start:end + 1]
    raise ContractError("最终输出里找不到 JSON 报告")


def run_task(task: dict, snaps: Snapshots, provider, max_turns: int = 8
             ) -> tuple[Report, Trace]:
    trace = Trace(task_id=task["id"], provider=getattr(provider, "name", "?"))
    box = Toolbox(snaps, trace)
    messages: list[dict] = [{"role": "user", "content": build_user_message(task)}]

    for _ in range(max_turns):
        resp: LLMResponse = provider.complete(SYSTEM, messages, TOOL_SPECS)
        trace.llm_turns += 1

        if resp.wants_tools:
            assistant_blocks = []
            if resp.text:
                assistant_blocks.append({"type": "text", "text": resp.text})
            for c in resp.tool_calls:
                assistant_blocks.append({"type": "tool_use", "id": c["id"],
                                         "name": c["name"], "input": c["input"]})
            messages.append({"role": "assistant", "content": assistant_blocks})

            results = []
            for c in resp.tool_calls:
                out = box.call(c["name"], c["input"])
                results.append({
                    "type": "tool_result", "tool_use_id": c["id"],
                    "content": json.dumps(out, ensure_ascii=False),
                    "is_error": "error" in out,
                })
            messages.append({"role": "user", "content": results})
            continue

        # 没有工具调用 = 模型认为可以出结论了
        trace.finished_at = time.time()
        report = Report.from_json(_extract_json(resp.text or ""))
        report.validate()
        return report, trace

    trace.finished_at = time.time()
    raise ContractError(f"超过最大轮次 {max_turns} 仍未产出报告")
