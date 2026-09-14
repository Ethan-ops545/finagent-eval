# -*- coding: utf-8 -*-
"""生成合成公司财报快照。

为什么用合成数据而不是真实财报，见 README「设计取舍」：
评测集需要**精确已知的真值**，而真实财报存在重述、口径调整与授权问题，
会把「模型错了」和「数据本身有歧义」混在一起，评测就不可判定了。

生成规则刻意保持内部一致（资产 = 负债 + 所有者权益，毛利 = 收入 - 成本 …），
这样确定性复算器的每一条断言都有唯一正确答案。

用法: python scripts/make_synthetic_data.py
"""
import json
import pathlib

OUT = pathlib.Path(__file__).resolve().parents[1] / "data" / "snapshots"

# 三家虚构公司。名字刻意不像任何真实公司，ticker 也不是真实代码。
COMPANIES = [
    {
        "ticker": "NOVA",
        "name": "Nova Grid Systems (虚构)",
        "sector": "电力设备",
        "base_revenue": 820_000.0,
        "growth": [0.18, 0.24, 0.31, 0.12],   # FY2022..FY2025 各年同比
        "gross_margin": [0.362, 0.371, 0.388, 0.375],
        "opex_ratio": [0.221, 0.215, 0.208, 0.219],
        "price": 42.10,
        "shares": 120_000.0,
    },
    {
        "ticker": "ORCA",
        "name": "Orca Marine Logistics (虚构)",
        "sector": "航运物流",
        "base_revenue": 1_450_000.0,
        "growth": [0.06, -0.04, 0.09, 0.05],
        "gross_margin": [0.191, 0.174, 0.203, 0.198],
        "opex_ratio": [0.128, 0.134, 0.126, 0.129],
        "price": 18.75,
        "shares": 310_000.0,
    },
    {
        "ticker": "PLNT",
        "name": "Plinth Materials (虚构)",
        "sector": "基础材料",
        "base_revenue": 640_000.0,
        "growth": [0.11, 0.15, 0.04, -0.02],
        "gross_margin": [0.284, 0.291, 0.276, 0.268],
        "opex_ratio": [0.176, 0.172, 0.181, 0.188],
        "price": 27.40,
        "shares": 95_000.0,
    },
]

PERIODS = ["FY2022", "FY2023", "FY2024", "FY2025"]

# 年报发布日：次年 2 月中旬。前视偏差检查靠这个字段。
PUBLISHED = {
    "FY2022": "2023-02-16",
    "FY2023": "2024-02-15",
    "FY2024": "2025-02-14",
    "FY2025": "2026-02-13",
}

# 市场数据的可见日期（快照日）。
MARKET_DATES = ["2024-04-15", "2025-04-15", "2026-04-15"]


def r2(x):
    """统一保留两位小数——避免浮点尾数让"复算结果不等于快照值"变成假阳性。"""
    return round(x + 0.0, 2)


def build(c):
    income, balance = {}, {}
    rev = c["base_revenue"]
    for i, p in enumerate(PERIODS):
        rev = rev * (1 + c["growth"][i])
        gm = c["gross_margin"][i]
        cogs = rev * (1 - gm)
        gross = rev - cogs
        opex = rev * c["opex_ratio"][i]
        op = gross - opex
        interest = rev * 0.011
        pretax = op - interest
        tax = pretax * 0.25
        net = pretax - tax
        income[p] = {
            "_published": PUBLISHED[p],
            "_unit": "千美元",
            "revenue": r2(rev),
            "cost_of_revenue": r2(cogs),
            "gross_profit": r2(gross),
            "operating_expenses": r2(opex),
            "operating_income": r2(op),
            "interest_expense": r2(interest),
            "pretax_income": r2(pretax),
            "income_tax": r2(tax),
            "net_income": r2(net),
        }
        # 资产负债表：先定各项，再用"权益 = 资产 - 负债"倒挤，保证恒等式严格成立。
        cash = rev * 0.14
        ar = rev * 0.19
        inv = cogs * 0.21
        ca = cash + ar + inv
        ppe = rev * 0.55
        ta = ca + ppe
        ap = cogs * 0.17
        std = rev * 0.08
        cl = ap + std
        ltd = rev * 0.26
        tl = cl + ltd
        balance[p] = {
            "_published": PUBLISHED[p],
            "_unit": "千美元",
            "cash_and_equivalents": r2(cash),
            "accounts_receivable": r2(ar),
            "inventory": r2(inv),
            "current_assets": r2(ca),
            "property_plant_equipment": r2(ppe),
            "total_assets": r2(ta),
            "accounts_payable": r2(ap),
            "short_term_debt": r2(std),
            "current_liabilities": r2(cl),
            "long_term_debt": r2(ltd),
            "total_liabilities": r2(tl),
            "total_equity": r2(r2(ta) - r2(tl)),  # 用已落盘的值倒挤，恒等式在快照层面成立
        }

    market = {}
    for j, d in enumerate(MARKET_DATES):
        market[d] = {
            "_published": d,
            "_unit": "美元 / 千股",
            "close_price": r2(c["price"] * (1 + 0.07 * j)),
            "shares_outstanding": r2(c["shares"] * (1 + 0.01 * j)),
        }

    return {
        "ticker": c["ticker"],
        "name": c["name"],
        "sector": c["sector"],
        "currency": "USD",
        "synthetic": True,
        "note": "合成数据，非真实公司财报。生成脚本见 scripts/make_synthetic_data.py",
        "statements": {"income": income, "balance": balance},
        "market": market,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for c in COMPANIES:
        snap = build(c)
        path = OUT / f"{c['ticker']}.json"
        path.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
        print("wrote", path)


if __name__ == "__main__":
    main()
