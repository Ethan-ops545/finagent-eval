# -*- coding: utf-8 -*-
"""把真实公司财报抓成同构的快照（可选）。

默认基准跑的是合成数据（理由见 README）。这个脚本用来验证同一套契约
在真实数据上也成立：字段名对齐后，Agent 与评测器一行都不用改。

需要 `pip install yfinance`。

    python scripts/fetch_real_snapshot.py AAPL MSFT --out data/real

注意：产出的快照只供本地试跑，不进基准集——真实财报存在重述与口径调整，
"正确答案"本身会变，不适合作为评测基准。
"""
from __future__ import annotations

import argparse
import json
import pathlib

FIELD_MAP_INCOME = {
    "revenue": ["Total Revenue"],
    "cost_of_revenue": ["Cost Of Revenue"],
    "gross_profit": ["Gross Profit"],
    "operating_expenses": ["Operating Expense"],
    "operating_income": ["Operating Income"],
    "interest_expense": ["Interest Expense"],
    "pretax_income": ["Pretax Income"],
    "income_tax": ["Tax Provision"],
    "net_income": ["Net Income"],
}
FIELD_MAP_BALANCE = {
    "cash_and_equivalents": ["Cash And Cash Equivalents"],
    "accounts_receivable": ["Accounts Receivable"],
    "inventory": ["Inventory"],
    "current_assets": ["Current Assets"],
    "property_plant_equipment": ["Net PPE"],
    "total_assets": ["Total Assets"],
    "accounts_payable": ["Accounts Payable"],
    "short_term_debt": ["Current Debt"],
    "current_liabilities": ["Current Liabilities"],
    "long_term_debt": ["Long Term Debt"],
    "total_liabilities": ["Total Liabilities Net Minority Interest"],
    "total_equity": ["Stockholders Equity"],
}


def pick(frame, names):
    for n in names:
        if n in frame.index:
            return frame.loc[n]
    return None


def convert(frame, mapping, published_lag_days=45):
    import pandas as pd
    out = {}
    if frame is None or frame.empty:
        return out
    for col in frame.columns:
        period = f"FY{col.year}"
        block = {
            "_published": str((col + pd.Timedelta(days=published_lag_days)).date()),
            "_unit": "美元",
        }
        for field, names in mapping.items():
            row = pick(frame, names)
            if row is None:
                continue
            val = row.get(col)
            if val is not None and val == val:      # 过滤 NaN
                block[field] = float(val)
        out[period] = block
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers", nargs="+")
    ap.add_argument("--out", default="data/real")
    args = ap.parse_args()

    try:
        import yfinance as yf
    except ImportError:
        raise SystemExit("需要先装依赖: pip install yfinance")

    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    for t in args.tickers:
        tk = yf.Ticker(t)
        info = tk.info or {}
        snap = {
            "ticker": t.upper(),
            "name": info.get("longName", t),
            "sector": info.get("sector", ""),
            "currency": info.get("currency", "USD"),
            "synthetic": False,
            "note": "来自 yfinance 的真实数据，仅供本地试跑，不作为评测基准。",
            "statements": {
                "income": convert(tk.income_stmt, FIELD_MAP_INCOME),
                "balance": convert(tk.balance_sheet, FIELD_MAP_BALANCE),
            },
            "market": {},
        }
        hist = tk.history(period="2y", interval="1mo")
        shares = info.get("sharesOutstanding")
        for ts, row in hist.iterrows():
            d = str(ts.date())
            snap["market"][d] = {
                "_published": d, "_unit": "美元 / 股",
                "close_price": float(row["Close"]),
                "shares_outstanding": float(shares) if shares else 0.0,
            }
        path = outdir / f"{t.upper()}.json"
        path.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
        print("wrote", path)


if __name__ == "__main__":
    main()
