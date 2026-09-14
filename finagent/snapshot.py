# -*- coding: utf-8 -*-
"""财报快照的读取与字段引用解析。

字段引用（field reference）是全项目的地址格式，Agent 和评测器都用它：

    NOVA.income.FY2024.revenue        利润表某期某字段
    NOVA.balance.FY2024.current_assets
    NOVA.market.2025-04-15.close_price

这是"证据可追溯"的物理基础：报告里的每个数字都必须挂着一串这样的地址，
评测器才有办法独立回到原始数据复算一遍。
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

SNAP_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "snapshots"


class RefError(KeyError):
    """字段引用格式错误或指向不存在的数据。"""


@dataclass(frozen=True)
class Resolved:
    ref: str
    value: float
    published: str      # 该数据的可见日期，用于前视偏差检查
    unit: str


class Snapshots:
    def __init__(self, directory: pathlib.Path | str = SNAP_DIR):
        self.dir = pathlib.Path(directory)
        self._cache: dict[str, dict] = {}

    def load(self, ticker: str) -> dict:
        t = ticker.upper()
        if t not in self._cache:
            path = self.dir / f"{t}.json"
            if not path.exists():
                raise RefError(f"没有 {t} 的快照: {path}")
            self._cache[t] = json.loads(path.read_text(encoding="utf-8"))
        return self._cache[t]

    def tickers(self) -> list[str]:
        return sorted(p.stem for p in self.dir.glob("*.json"))

    def period_block(self, ticker: str, statement: str, period: str) -> dict:
        snap = self.load(ticker)
        if statement == "market":
            block = snap.get("market", {}).get(period)
        else:
            block = snap.get("statements", {}).get(statement, {}).get(period)
        if block is None:
            raise RefError(f"没有这一期数据: {ticker}.{statement}.{period}")
        return block

    def resolve(self, ref: str) -> Resolved:
        """把一条字段引用解析成 (值, 可见日期, 单位)。"""
        parts = ref.split(".")
        if len(parts) != 4:
            raise RefError(f"字段引用格式应为 TICKER.statement.PERIOD.field，收到: {ref}")
        ticker, statement, period, field = parts
        if statement not in ("income", "balance", "market"):
            raise RefError(f"未知报表类型: {statement}")
        if field.startswith("_"):
            raise RefError(f"元数据字段不可作为结论输入: {field}")
        block = self.period_block(ticker, statement, period)
        if field not in block:
            raise RefError(f"字段不存在: {ref}")
        return Resolved(
            ref=ref,
            value=float(block[field]),
            published=str(block.get("_published", period)),
            unit=str(block.get("_unit", "")),
        )

    def public_fields(self, ticker: str, statement: str, period: str) -> dict:
        """去掉下划线元数据后的可见字段，工具返回给 Agent 的就是这个。"""
        return {k: v for k, v in self.period_block(ticker, statement, period).items()
                if not k.startswith("_")}
