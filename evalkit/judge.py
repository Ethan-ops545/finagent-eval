# -*- coding: utf-8 -*-
"""判断项打分 —— 只处理确定性评测**管不了**的那部分。

分界线很清楚：一条结论只要能回到原始数据复算，就绝不拿到这里来打分。
拿模型去评判一个能被算出来的数，是把确定的事情变回不确定的事情。

留给这里的只有表述质量：结论有没有说清依据、有没有超出数据作推断、
有没有把增长与盈利放在一起看。这些没有唯一答案，只能按 rubric 逐条给分。

两种实现：
  HeuristicJudge —— 关键词规则。**刻意做得很粗糙**，它的作用是让整条流水线
                    在没有 API key 时可测，不是要假装自己会评价文章。
  LLMJudge       —— 真让大模型按 rubric 打分，输出结构化 JSON。

无论哪种，打分结果都单独成列，绝不并进确定性得分里做成一个"总分"——
一旦合并，一个漂亮的表述就能把一个算错的数字盖过去。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict


@dataclass
class RubricScore:
    criterion: str
    passed: bool
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


FORWARD_LOOKING = ("预计", "预期将", "明年将", "有望", "料将")
GROWTH_WORDS = ("增长", "增速", "同比", "下滑")
PROFIT_WORDS = ("毛利", "净利", "盈利")

# 偿债能力的"定论词"。说出这些词，就是在下一个**跨行业不成立**的判断：
# 利息保障 7 倍，在重资产高杠杆的航运业可能很正常，在轻资产行业可能就偏低。
# 所以这类词本身不是错，**缺少语境**才是错。
VERDICT_WORDS = ("充足", "安全", "健康", "无虞", "无风险", "危险", "堪忧", "紧张", "承压")
CONTEXT_WORDS = ("行业", "同业", "可比", "周期", "重资产", "轻资产",
                 "债务期限", "资产结构", "语境", "参照", "基准")
COVERAGE_WORDS = ("利息保障", "利息覆盖", "EBIT")
LEVERAGE_WORDS = ("负债率", "负债/资产", "杠杆", "总负债")


class HeuristicJudge:
    name = "heuristic"

    def score(self, report, task) -> list[RubricScore]:
        s = report.summary
        out = []
        for criterion in task.get("rubric", []):
            if "增长" in criterion and "盈利" in criterion:
                ok = (any(w in s for w in GROWTH_WORDS)
                      and any(w in s for w in PROFIT_WORDS))
                out.append(RubricScore(criterion, ok,
                                       "同时提到增长与盈利" if ok else "只提到其中一条线"))
            elif "可见日" in criterion or "期间" in criterion:
                ok = task["as_of"] in s and task["fiscal_year"] in s
                out.append(RubricScore(criterion, ok,
                                       "写明了期间与信息可见日" if ok else "未写明期间或可见日"))
            elif "推断" in criterion or "预测" in criterion:
                hits = [w for w in FORWARD_LOOKING if w in s]
                out.append(RubricScore(criterion, not hits,
                                       "未见前瞻性断言" if not hits
                                       else f"出现前瞻性措辞: {hits}"))
            elif "行业语境" in criterion:
                # 关键不是"有没有下判断"，而是"下判断时有没有交代它依赖什么"。
                verdict = [w for w in VERDICT_WORDS if w in s]
                context = [w for w in CONTEXT_WORDS if w in s]
                if not verdict:
                    out.append(RubricScore(criterion, True, "未下无语境的定论"))
                elif context:
                    out.append(RubricScore(criterion, True,
                                           f"下了判断（{verdict[0]}）但交代了语境（{context[0]}）"))
                else:
                    out.append(RubricScore(criterion, False,
                                           f"下了定论（{verdict}）却未交代行业语境或比较基准"))
            elif "利息保障" in criterion and "负债率" in criterion:
                cov = any(w in s for w in COVERAGE_WORDS)
                lev = any(w in s for w in LEVERAGE_WORDS)
                missing = ([] if cov else ["盈利端（利息保障）"]) + ([] if lev else ["资产负债端（负债率）"])
                out.append(RubricScore(criterion, cov and lev,
                                       "两端都用到了" if cov and lev
                                       else f"只看了一端，缺: {'、'.join(missing)}"))
            else:
                out.append(RubricScore(criterion, True, "启发式规则未覆盖，默认通过"))
        return out


JUDGE_PROMPT = """你在按 rubric 给一段金融研究结论打分。只判断表述质量，不要核对数字——
数字已经由确定性复算器单独检查过了，不属于你的职责。

结论原文：
{summary}

逐条判断下列标准，对每条给出 passed(true/false) 与一句话理由。
只输出 JSON 数组，不要别的内容：
[{{"criterion": "...", "passed": true, "reason": "..."}}]

标准：
{rubric}
"""


class LLMJudge:
    name = "llm"

    def __init__(self, provider):
        self.provider = provider

    def score(self, report, task) -> list[RubricScore]:
        rubric = task.get("rubric", [])
        if not rubric:
            return []
        prompt = JUDGE_PROMPT.format(
            summary=report.summary,
            rubric="\n".join(f"- {r}" for r in rubric))
        resp = self.provider.complete("你是一个严格的评审。",
                                      [{"role": "user", "content": prompt}], [])
        text = resp.text or "[]"
        m = re.search(r"\[.*\]", text, re.S)
        try:
            data = json.loads(m.group(0) if m else text)
        except json.JSONDecodeError:
            return [RubricScore(r, False, "评审输出无法解析") for r in rubric]
        return [RubricScore(str(d.get("criterion", "")), bool(d.get("passed")),
                            str(d.get("reason", ""))) for d in data]
