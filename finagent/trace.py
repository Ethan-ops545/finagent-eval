# -*- coding: utf-8 -*-
"""全链路追踪。

每一次工具调用都落一条结构化事件，评测器靠它回答两个问题：

  1. 这条结论引用的数据，Agent **到底取过没有**？
     没取过却写进报告 = 凭空捏造（ungrounded），这是最危险的一类幻觉——
     因为数值可能恰好是对的，只靠对答案根本发现不了。
  2. 工具调用本身对不对？（参数错、调了不该调的期间、重复调用）
"""
from __future__ import annotations

import json
import pathlib
import time
from dataclasses import dataclass, field, asdict


@dataclass
class ToolCall:
    step: int
    tool: str
    args: dict
    ok: bool
    latency_ms: float
    error: str | None = None
    # 这次调用让 Agent "看到"了哪些字段引用。接地性检查全靠它。
    exposed_refs: list[str] = field(default_factory=list)


@dataclass
class Trace:
    task_id: str
    provider: str
    calls: list[ToolCall] = field(default_factory=list)
    llm_turns: int = 0
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    def add(self, call: ToolCall) -> None:
        self.calls.append(call)

    def exposed(self) -> set[str]:
        """Agent 在本次运行中真正看到过的全部字段引用。"""
        out: set[str] = set()
        for c in self.calls:
            if c.ok:
                out.update(c.exposed_refs)
        return out

    def tool_error_count(self) -> int:
        return sum(1 for c in self.calls if not c.ok)

    def to_dict(self) -> dict:
        return asdict(self)

    def write_jsonl(self, path: pathlib.Path | str) -> None:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"event": "run_start", "task_id": self.task_id,
                                "provider": self.provider, "ts": self.started_at},
                               ensure_ascii=False) + "\n")
            for c in self.calls:
                f.write(json.dumps({"event": "tool_call", **asdict(c)},
                                   ensure_ascii=False) + "\n")
            f.write(json.dumps({"event": "run_end", "llm_turns": self.llm_turns,
                                "tool_calls": len(self.calls),
                                "tool_errors": self.tool_error_count(),
                                "ts": self.finished_at}, ensure_ascii=False) + "\n")
