from __future__ import annotations
from dataclasses import dataclass

@dataclass
class BasketSnapshot:
    prices: dict[str, float]
    ref_prices: dict[str, float]
    basket_value: float
    ref_value: float
    change_pct: float

def compute_equal_weight_basket(prices: dict[str, float], ref_prices: dict[str, float]) -> BasketSnapshot:
    tickers = [t for t in ref_prices.keys() if t in prices]
    if not tickers:
        return BasketSnapshot(prices, ref_prices, 0.0, 0.0, 0.0)

    # equal-weight "index": average of price relatives (P / Pref)
    relatives = [(prices[t] / ref_prices[t]) for t in tickers if ref_prices[t] > 0]
    if not relatives:
        return BasketSnapshot(prices, ref_prices, 0.0, 0.0, 0.0)

    ref_value = 1.0
    basket_value = sum(relatives) / len(relatives)
    change_pct = (basket_value - ref_value) / ref_value
    return BasketSnapshot(prices, ref_prices, basket_value, ref_value, change_pct)

def classify_trigger(change_pct: float, levels: list[tuple[str, float]]) -> tuple[str, float] | None:
    """
    levels: [(kind, level_pct), ...] e.g. [("TP", 0.03), ("SL", -0.05)]
    We fire the "closest crossed" level:
      - for TP: highest level <= change_pct
      - for SL: lowest level >= change_pct (more negative)
    """
    tp = sorted([lv for lv in levels if lv[0] == "TP"], key=lambda x: x[1])
    sl = sorted([lv for lv in levels if lv[0] == "SL"], key=lambda x: x[1])  # e.g. -0.10, -0.07, -0.05

    hit_tp = [lv for lv in tp if change_pct >= lv[1]]
    hit_sl = [lv for lv in sl if change_pct <= lv[1]]

    if hit_tp:
        return hit_tp[-1]  # highest passed take-profit
    if hit_sl:
        return hit_sl[0]   # most negative threshold that has been crossed? (sorted ascending)
    return None
