from __future__ import annotations
import yfinance as yf
import requests
import pandas as pd
from io import StringIO
from datetime import date

# Reuse connections + set a user agent (often helps)
_SESSION = requests.Session()
_HEADERS = {"User-Agent": "MayfairCapitalBasket/1.0"}

import yfinance as yf

def fetch_latest_prices_yf(tickers: list[str]) -> dict[str, float]:
    """
    Yahoo fallback. Returns dict keyed by UPPERCASE original tickers.
    Converts BRK.B -> BRK-B for Yahoo lookup.
    """
    out: dict[str, float] = {}
    for t in tickers:
        key = t.strip().upper()
        yf_symbol = key.replace(".", "-")  # Yahoo uses BRK-B etc.
        try:
            tk = yf.Ticker(yf_symbol)
            price = tk.fast_info.get("last_price")
            if price is None:
                # fallback: try recent close
                hist = tk.history(period="1d")
                if not hist.empty:
                    price = float(hist["Close"].iloc[-1])
            if price is not None:
                out[key] = float(price)
        except Exception:
            # silent fail; caller decides what to do
            pass
    return out


def fetch_latest_prices_with_fallback(
    tickers: list[str],
    timeout: int = 10,
    debug: bool = False,
) -> dict[str, float]:
    """
    Primary: Stooq.
    Fallback: Yahoo Finance (yfinance) for missing tickers.
    Returns dict keyed by UPPERCASE tickers.
    """
    # 1) Stooq first
    stooq_prices = fetch_latest_prices(tickers, timeout=timeout, debug=debug)

    # normalize keys (in case your stooq fetch returns mixed keys)
    stooq_prices = {k.strip().upper(): float(v) for k, v in stooq_prices.items()}

    missing = [t.strip().upper() for t in tickers if t.strip().upper() not in stooq_prices]
    if not missing:
        return stooq_prices

    # 2) Yahoo only for missing
    yf_prices = fetch_latest_prices_yf(missing)

    if debug:
        print(f"[FALLBACK] Missing from Stooq: {missing}")
        print(f"[FALLBACK] Yahoo filled: {list(yf_prices.keys())}")

    merged = dict(stooq_prices)
    merged.update(yf_prices)
    return merged


def _to_stooq_symbol(ticker: str) -> str:
    t = ticker.strip().lower()
    if not t.endswith(".us"):
        t = f"{t}.us"
    return t


# Stooq CSV endpoint: https://stooq.com/q/l/?s=aapl.us&i=1m
# We'll request "latest" via 1m and take the last row.
def fetch_latest_price(ticker: str, timeout: int = 10, debug: bool = False) -> float | None:
    try:
        sym = _to_stooq_symbol(ticker)
        url = f"https://stooq.com/q/l/?s={sym}&i=1m"
        r = _SESSION.get(url, timeout=timeout, headers=_HEADERS)
        r.raise_for_status()

        # --- DEBUG: print the raw CSV returned by Stooq (first ~300 chars) ---
        if debug:
            print(f"[STOOQ RAW] {sym} -> {r.text[:300].replace(chr(10), ' | ')}")

        df = pd.read_csv(StringIO(r.text))
        if df.empty or "Close" not in df.columns:
            return None

        price = df["Close"].iloc[-1]
        if pd.isna(price):
            return None
        return float(price)
    except Exception as e:
        if debug:
            print(f"[STOOQ ERROR] {ticker} -> {e}")
        return None


def fetch_latest_prices(tickers: list[str], timeout: int = 10, debug: bool = False) -> dict[str, float]:
    out: dict[str, float] = {}
    for t in tickers:
        key = t.strip().upper()  # normalize keys
        p = fetch_latest_price(key, timeout=timeout, debug=debug)
        if p is not None:
            out[key] = p
    return out


def fetch_daily_history(ticker: str, timeout: int = 10, debug: bool = False) -> pd.DataFrame:
    try:
        sym = _to_stooq_symbol(ticker)
        url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
        r = _SESSION.get(url, timeout=timeout, headers=_HEADERS)
        r.raise_for_status()

        if debug:
            print(f"[STOOQ RAW DAILY] {sym} -> {r.text[:300].replace(chr(10), ' | ')}")

        df = pd.read_csv(StringIO(r.text))
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
        return df
    except Exception as e:
        if debug:
            print(f"[STOOQ ERROR DAILY] {ticker} -> {e}")
        return pd.DataFrame()


def first_trading_day_close_for_month(ticker: str, month_start: date, timeout: int = 10, debug: bool = False) -> float | None:
    df = fetch_daily_history(ticker, timeout=timeout, debug=debug)
    if df.empty or "Date" not in df.columns or "Close" not in df.columns:
        return None

    df2 = df[df["Date"] >= month_start].sort_values("Date", ascending=True)
    if df2.empty:
        return None

    val = df2.iloc[0]["Close"]
    if pd.isna(val):
        return None
    return float(val)
