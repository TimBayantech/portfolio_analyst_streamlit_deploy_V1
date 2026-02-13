from __future__ import annotations

from datetime import datetime, UTC
from datetime import date as dt_date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from dotenv import load_dotenv
from sqlalchemy import select, delete

from app.db import SessionLocal, engine
from app.models import Basket, Holding, MonthReference, TriggerLevel, TriggerEvent
from app.stooq import fetch_latest_prices_with_fallback
from app.basket import compute_equal_weight_basket, classify_trigger
from app.emailer import send_trigger_email
from app.logging_setup import get_logger
import plotly.graph_objects as go
import yfinance as yf
import math
import streamlit.components.v1 as components

BELL_PATH = Path("assets/bell.wav")

load_dotenv()
logger = get_logger()

# --- DB init ---
from app.db import Base  # noqa: E402
Base.metadata.create_all(bind=engine)

DEFAULT_TPS = [0.03, 0.04, 0.06]
DEFAULT_SLS = [-0.05, -0.07, -0.10]

@st.cache_data(ttl=300)
def fetch_daily_closes_yf(tickers: list[str], start: dt_date, end: dt_date) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by date with columns per ticker.
    Uses yfinance. End is exclusive-ish, yfinance usually treats as end date.
    """
    if not tickers:
        return pd.DataFrame()

    # yfinance expects strings; BRK.B => BRK-B
    yft = [t.replace(".", "-") for t in tickers]

    df = yf.download(
        tickers=" ".join(yft),
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),  # ensure we include end day
        progress=False,
        auto_adjust=False,
        group_by="ticker",
        threads=True
    )

    if df.empty:
        return pd.DataFrame()

    # Normalize to a simple "Close" wide frame
    if isinstance(df.columns, pd.MultiIndex):
        # MultiIndex: (field, ticker) or (ticker, field) depending on yfinance
        # yfinance commonly returns columns like ('Close', 'AAPL') in one layout
        if "Close" in df.columns.get_level_values(0):
            closes = df["Close"].copy()
        else:
            # alternate layout: (ticker, field)
            closes = pd.DataFrame({t: df[t]["Close"] for t in df.columns.get_level_values(0).unique() if "Close" in df[t]})
    else:
        # Single ticker case returns Series-like columns
        if "Close" in df.columns:
            closes = df[["Close"]].rename(columns={"Close": tickers[0]})
        else:
            return pd.DataFrame()

    # Convert column names back from yfinance symbols to your tickers where possible
    closes.columns = [c.replace("-", ".") for c in closes.columns]
    closes.index = pd.to_datetime(closes.index).date
    return closes.sort_index()


def compute_portfolio_mtd_series(closes: pd.DataFrame, ref_map: dict[str, float]) -> pd.Series:
    """
    Equal-weight MTD return % series using month-start ref prices (ref_map).
    For each day: mean((close/ref)-1) across tickers available that day.
    """
    if closes.empty:
        return pd.Series(dtype=float)

    # Keep only tickers that have a valid ref
    valid_cols = [c for c in closes.columns if c in ref_map and ref_map.get(c, 0) > 0]
    if not valid_cols:
        return pd.Series(dtype=float)

    df = closes[valid_cols].copy()

    # Compute daily returns vs month-start ref
    for c in valid_cols:
        df[c] = (df[c].astype(float) / float(ref_map[c])) - 1.0

    # Equal-weight mean across tickers present (skip NaNs)
    port = df.mean(axis=1, skipna=True) * 100.0
    port.name = "Portfolio"
    return port


@st.cache_data(ttl=300)
def fetch_index_mtd_series(symbol: str, start: dt_date, end: dt_date, name: str) -> pd.Series:
    """
    Fetch index close series then convert to MTD return % anchored at 0 on first available date.
    Uses Ticker().history() which is often more reliable than yf.download() for indices.
    """
    try:
        tk = yf.Ticker(symbol)
        hist = tk.history(
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat()
        )
        if hist is None or hist.empty or "Close" not in hist.columns:
            return pd.Series(dtype=float, name=name)

        s = hist["Close"].copy()
        s.index = pd.to_datetime(s.index).date
        s = s.sort_index()

        first = float(s.iloc[0])
        if first <= 0:
            return pd.Series(dtype=float, name=name)

        out = (s / first - 1.0) * 100.0
        out.name = name
        return out
    except Exception:
        return pd.Series(dtype=float, name=name)

# -------------------------
# Tiles UI helpers (NEW)
# -------------------------
def inject_tile_css():
    st.markdown("""
    <style>
    .tile {
        padding: 14px;
        border-radius: 14px;
        height: 88px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        font-family: ui-sans-serif, system-ui, -apple-system;
        font-weight: 600;
        box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    }
    .bg-pos { background: rgba(34, 197, 94, 0.18); border: 1px solid rgba(34, 197, 94, 0.35); }
    .bg-neg { background: rgba(239, 68, 68, 0.18); border: 1px solid rgba(239, 68, 68, 0.35); }
    .bg-flat{ background: rgba(148, 163, 184, 0.18); border: 1px solid rgba(148, 163, 184, 0.35); }
    .ticker { font-size: 18px; letter-spacing: 0.5px; }
    .ret { font-size: 16px; opacity: 0.95; margin-top: 4px; }
    </style>
    """, unsafe_allow_html=True)

def bg_class(ret: float, eps: float = 0.0001) -> str:
    if ret > eps:
        return "bg-pos"
    if ret < -eps:
        return "bg-neg"
    return "bg-flat"


def render_tiles(title: str, items: list[dict], cols_per_row: int = 6, rows: int = 4):
    st.subheader(title)
    items_sorted = sorted(items, key=lambda x: x.get("ticker", ""))

    total_slots = cols_per_row * rows
    padded = items_sorted[:total_slots] + [None] * max(0, total_slots - len(items_sorted))

    idx = 0
    for _ in range(rows):
        cols = st.columns(cols_per_row)
        for col in cols:
            item = padded[idx]
            idx += 1
            with col:
                if item is None:
                    st.markdown('<div class="tile bg-flat"></div>', unsafe_allow_html=True)
                else:
                    cls = bg_class(float(item.get("return", 0.0)))
                    ret_str = f'{float(item.get("return", 0.0)):+.2f}%'
                    st.markdown(
                        f"""
                        <div class="tile {cls}">
                            <div class="ticker">{item.get("ticker","")}</div>
                            <div class="ret">{ret_str}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

def render_portfolio_tiles(portfolio_items: list[dict]):
    inject_tile_css()

    nasdaq = [p for p in portfolio_items if p.get("source") == "NASDAQ"]
    dow = [p for p in portfolio_items if p.get("source") == "DOW"]

    nas_rows = max(1, math.ceil(len(nasdaq) / 6))
    dow_rows = max(1, math.ceil(len(dow) / 6))

    render_tiles("NASDAQ", nasdaq, cols_per_row=6, rows=nas_rows)
    st.divider()
    render_tiles("DOW", dow, cols_per_row=6, rows=dow_rows)

def classify_exchange_simple(ticker: str) -> str:
    """
    (NEW) Simple heuristic because your holdings table doesn't store exchange.
    If you later add Holding.source, replace this with that field.
    """
    # Common Dow tickers (subset); expand as you like
    dow_set = {
        "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "DOW",
        "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM",
        "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "V", "VZ", "WBA", "WMT"
    }
    t = ticker.strip().upper()
    return "DOW" if t in dow_set else "NASDAQ"


def build_tile_items(holdings: list[Holding], ref_map: dict, prices: dict) -> list[dict]:
    """
    Build items for the tiles grid from your existing data.
    return is expressed as percentage number (e.g., +3.42).
    """
    items: list[dict] = []
    for h in holdings:
        if not h.active:
            continue
        t = h.ticker
        ref_p = ref_map.get(t, None)
        cur_p = prices.get(t, None)
        if ref_p is None or cur_p is None or float(ref_p) <= 0:
            continue

        ret = (float(cur_p) / float(ref_p) - 1.0) * 100.0
        items.append(
            {
                "ticker": t,
                "source": classify_exchange_simple(t),
                "return": float(ret),
            }
        )
    return items


def month_key_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m")


def ensure_basket(session) -> Basket:
    b = session.execute(select(Basket).where(Basket.name == "Mayfair Capital Basket")).scalar_one_or_none()
    if not b:
        b = Basket(name="Mayfair Capital Basket", poll_seconds=60)
        session.add(b)
        session.commit()
        session.refresh(b)
        for lv in DEFAULT_TPS:
            session.add(TriggerLevel(basket_id=b.id, kind="TP", level_pct=lv, enabled=True))
        for lv in DEFAULT_SLS:
            session.add(TriggerLevel(basket_id=b.id, kind="SL", level_pct=lv, enabled=True))
        session.commit()
    return b


def parse_upload(file) -> list[str]:
    """Parse uploaded CSV/XLSX and return up to 24 UNIQUE tickers (uppercased)."""
    if file is None:
        return []
    name = file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    cols = [str(c).strip().lower() for c in df.columns]
    if "ticker" in cols:
        col = df.columns[cols.index("ticker")]
    elif "symbol" in cols:
        col = df.columns[cols.index("symbol")]
    else:
        col = df.columns[0]

    tickers = [str(x).strip().upper() for x in df[col].dropna().tolist()]
    tickers = [t for t in tickers if t]

    seen = set()
    out: list[str] = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            out.append(t)

    return out[:24]


# ---------------------------
# Month-start reference fetch
# ---------------------------
def fetch_month_start_ref_price(ticker: str, month_start: dt_date) -> tuple[float | None, str]:
    """
    Return (ref_price, ref_source). NEVER uses live price.
    ref_source values:
      - STOOQ_MONTH_START
      - YF_MONTH_START
      - NONE
    """
    t = ticker.strip().upper()

    # 1) Stooq: first trading day close on/after month_start
    try:
        from app.stooq import first_trading_day_close_for_month
        ref = first_trading_day_close_for_month(t, month_start)
        if ref is not None:
            return float(ref), "STOOQ_MONTH_START"
    except Exception:
        pass

    # 2) Yahoo: first available close from a small window starting month_start
    try:
        import yfinance as yf
        sym = t.replace(".", "-")  # e.g., BRK.B -> BRK-B
        tk = yf.Ticker(sym)
        end = month_start + timedelta(days=10)
        hist = tk.history(start=month_start.isoformat(), end=end.isoformat())
        if not hist.empty and "Close" in hist.columns:
            return float(hist["Close"].iloc[0]), "YF_MONTH_START"
    except Exception:
        pass

    return None, "NONE"

def save_holdings_and_reference(session, basket: Basket, tickers: list[str]):
    """Replace holdings and ensure this month's reference prices exist. NEVER uses live price backfill."""
    session.execute(delete(Holding).where(Holding.basket_id == basket.id))
    for t in tickers:
        session.add(Holding(basket_id=basket.id, ticker=t.strip().upper(), weight=1.0, active=True))
    session.commit()

    mkey = month_key_now()
    existing = session.execute(
        select(MonthReference).where(
            MonthReference.basket_id == basket.id,
            MonthReference.month_key == mkey,
        )
    ).scalars().all()
    if existing:
        return

    today = dt_date.today()
    month_start = dt_date(today.year, today.month, 1)

    for t in tickers:
        t_norm = t.strip().upper()
        ref, _source = fetch_month_start_ref_price(t_norm, month_start)
        if ref is not None:
            session.add(
                MonthReference(
                    basket_id=basket.id,
                    month_key=mkey,
                    ticker=t_norm,
                    ref_price=float(ref),
                )
            )
    session.commit()


def get_state(session, basket: Basket):
    holdings = session.execute(
        select(Holding).where(Holding.basket_id == basket.id).order_by(Holding.ticker)
    ).scalars().all()

    active_tickers = [h.ticker for h in holdings if h.active]

    mkey = month_key_now()
    refs = session.execute(
        select(MonthReference).where(
            MonthReference.basket_id == basket.id,
            MonthReference.month_key == mkey,
        )
    ).scalars().all()
    ref_map = {r.ticker: r.ref_price for r in refs}

    triggers = session.execute(
        select(TriggerLevel).where(
            TriggerLevel.basket_id == basket.id,
            TriggerLevel.enabled == True,  # noqa: E712
        )
    ).scalars().all()
    levels = [(t.kind, float(t.level_pct)) for t in triggers]
    return holdings, active_tickers, ref_map, levels, triggers


def already_triggered(session, basket_id: int, month_key: str, kind: str, level_pct: float) -> bool:
    ev = session.execute(
        select(TriggerEvent).where(
            TriggerEvent.basket_id == basket_id,
            TriggerEvent.month_key == month_key,
            TriggerEvent.kind == kind,
            TriggerEvent.triggered_level_pct == level_pct,
        )
    ).scalar_one_or_none()
    return ev is not None


def record_trigger(session, basket_id: int, month_key: str, kind: str, level_pct: float, change_pct: float, note: str):
    session.add(
        TriggerEvent(
            basket_id=basket_id,
            month_key=month_key,
            kind=kind,
            triggered_level_pct=level_pct,
            basket_change_pct=change_pct,
            note=note[:300],
        )
    )
    session.commit()


def ensure_reference_prices(session, basket: Basket, tickers: list[str]) -> dict[str, str]:
    """
    Ensure a MonthReference exists for each ACTIVE ticker (current month).
    NEVER uses live price backfill.
    Returns ref_source map keyed by ticker for refs that exist or were created this run:
      - EXISTING
      - STOOQ_MONTH_START
      - YF_MONTH_START
    """
    mkey = month_key_now()

    existing_refs = session.execute(
        select(MonthReference).where(
            MonthReference.basket_id == basket.id,
            MonthReference.month_key == mkey,
        )
    ).scalars().all()

    existing_set = {r.ticker for r in existing_refs}
    ref_source: dict[str, str] = {t: "EXISTING" for t in existing_set}

    today = dt_date.today()
    month_start = dt_date(today.year, today.month, 1)

    created_any = False
    for t in tickers:
        t = t.strip().upper()
        if t in existing_set:
            continue

        ref, source = fetch_month_start_ref_price(t, month_start)
        if ref is not None:
            session.add(
                MonthReference(
                    basket_id=basket.id,
                    month_key=mkey,
                    ticker=t,
                    ref_price=float(ref),
                )
            )
            ref_source[t] = source
            created_any = True

    if created_any:
        session.commit()

    return ref_source


# --- UI ---
st.set_page_config(page_title="Mayfair Capital Basket", layout="wide")

st.markdown(
    """
<style>
.small-muted { color: rgba(0,0,0,0.55); font-size: 0.9rem; }

/* Make the main st.metric value larger */
div[data-testid="stMetricValue"] {
    font-size: 3.0rem;
    line-height: 1.1;
}

</style>
""",
    unsafe_allow_html=True,
)

with SessionLocal() as session:
    basket = ensure_basket(session)

left, rhs = st.columns([1.15, 0.85], gap="large")

# Load state once
with SessionLocal() as session:
    basket = ensure_basket(session)
    holdings, active_tickers, ref_map, levels, triggers = get_state(session, basket)

recipient_list: list[str] = [x.strip() for x in st.session_state.get("recipients_text", "").splitlines() if x.strip()]

# -------------------------
# LEFT: basket headline + tiles + triggers
# -------------------------
with left:
    st.title("Mayfair Capital - Portfolio Dashboard")

    if not active_tickers:
        st.warning("No tickers loaded yet. Upload a CSV/Excel file in Controls.")
        st.stop()

    mkey = month_key_now()

    # -------------------------
    # Create placeholders FIRST (so we can update them during fetch)
    # -------------------------
    metric_col, status_col = st.columns([0.70, 0.30], vertical_alignment="center")

    with metric_col:
        metric_placeholder = st.empty()

    with status_col:
        status_placeholder = st.empty()
        countdown_placeholder = st.empty()

    # -------------------------
    # Ensure refs exist for active tickers (month-start only; NO live backfill)
    # -------------------------
    with SessionLocal() as session:
        basket = ensure_basket(session)
        ref_source_map = ensure_reference_prices(session, basket, active_tickers)

    # Reload state so ref_map includes any newly created refs
    with SessionLocal() as session:
        basket = ensure_basket(session)
        holdings, active_tickers, ref_map, levels, triggers = get_state(session, basket)

    # -------------------------
    # Fetch prices (normal cadence); countdown is JS-only (no reruns needed)
    # -------------------------
    now = datetime.now(UTC)

    last_fetch = st.session_state.get("last_fetch_utc", None)
    cached_prices = st.session_state.get("cached_prices", None)

    elapsed = (now - last_fetch).total_seconds() if last_fetch else None
    due = (last_fetch is None) or (elapsed is not None and elapsed >= basket.poll_seconds)

    if due:
        status_placeholder.info("⏳ Updating from STOOQ/YAHOO")

        #with st.spinner("Fetching latest prices from STOOQ/YAHOO..."):
        prices = fetch_latest_prices_with_fallback(active_tickers, debug=False)

        st.session_state["cached_prices"] = prices
        st.session_state["last_fetch_utc"] = now

        status_placeholder.success("✅ Prices updated")
        last_fetch = now  # update local for countdown below

        f"Price refresh completed at {now.isoformat()}"

        if BELL_PATH.exists():
            st.audio(str(BELL_PATH), autoplay=True)

    else:
        prices = cached_prices if isinstance(cached_prices, dict) else {}
        status_placeholder.caption("🟢 Live (cached)")
    # -------------------------
    # Compute basket AFTER prices exist
    # -------------------------
    snap = compute_equal_weight_basket(prices, ref_map)
    tracked_active = [t for t in active_tickers if (t in ref_map and t in prices and ref_map.get(t, 0) > 0)]

    metric_placeholder.metric(
        label="Mayfair Capital Basket (Month-to-date)",
        value=f"{snap.change_pct * 100:.2f}%",
        delta=f"{snap.change_pct * 100:.2f}%",
    )

    st.caption(
        f"Month reference: {mkey} · "
        f"Active holdings: {len(active_tickers)} · "
        f"Tracked (ref+live): {len(tracked_active)}"
    )

    # -------------------------
    # Plotly chart (MTD Portfolio vs DOW vs NASDAQ)
    # -------------------------
    st.markdown("### Month-to-date performance (Portfolio vs DOW vs NASDAQ)")

    today = dt_date.today()
    month_start = dt_date(today.year, today.month, 1)

    tickers_for_history = [t for t in active_tickers if t in ref_map and ref_map.get(t, 0) > 0]

    closes = fetch_daily_closes_yf(tickers_for_history, start=month_start, end=today)
    portfolio_mtd = compute_portfolio_mtd_series(closes, ref_map)

    dow_mtd = fetch_index_mtd_series("^DJI", start=month_start, end=today, name="DOW (^DJI)")
    nas_mtd = fetch_index_mtd_series("^IXIC", start=month_start, end=today, name="NASDAQ (^IXIC)")

    if dow_mtd.empty or nas_mtd.empty:
        st.caption(f"Index data missing? DOW points: {len(dow_mtd)} · NASDAQ points: {len(nas_mtd)}")

    fig = go.Figure()

    if not portfolio_mtd.empty:
        fig.add_trace(go.Scatter(
            x=list(portfolio_mtd.index),
            y=portfolio_mtd.values,
            mode="lines",
            name="Portfolio (equal-weight)"
        ))

    if not dow_mtd.empty:
        fig.add_trace(go.Scatter(
            x=list(dow_mtd.index),
            y=dow_mtd.values,
            mode="lines",
            name="DOW (^DJI)"
        ))

    if not nas_mtd.empty:
        fig.add_trace(go.Scatter(
            x=list(nas_mtd.index),
            y=nas_mtd.values,
            mode="lines",
            name="NASDAQ (^IXIC)"
        ))

    fig.add_hline(y=0, line_width=1)

    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Date",
        yaxis_title="Return (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # Triggers
    # -------------------------
    hit = classify_trigger(snap.change_pct, levels)
    if hit is not None:
        kind, level_pct = hit
        st.warning(f"{kind} trigger crossed at {level_pct * 100:.2f}%")

        with SessionLocal() as session:
            basket = ensure_basket(session)
            if not already_triggered(session, basket.id, mkey, kind, level_pct):
                note = (
                    f"Basket change {snap.change_pct * 100:.2f}% "
                    f"crossed {level_pct * 100:.2f}% ({kind})."
                )
                record_trigger(session, basket.id, mkey, kind, level_pct, snap.change_pct, note)
                logger.info(note)

                if recipient_list:
                    subject = f"Mayfair Capital Basket trigger: {kind} {level_pct * 100:.2f}%"
                    content = (
                        f"{note}\n"
                        f"Month: {mkey}\n"
                        f"Time (UTC): {datetime.now(UTC).isoformat()}Z"
                    )
                    ok, msg = send_trigger_email(recipient_list, subject, content)
                    logger.info(f"Email: {ok} | {msg}")

                if BELL_PATH.exists():
                    st.audio(str(BELL_PATH), autoplay=True)

    st.caption(f"Auto-refresh every {basket.poll_seconds}s")
    st_autorefresh(interval=int(basket.poll_seconds * 1000), key="auto_refresh")

# -------------------------
# RHS: Per-ticker contribution table + Controls below fold
# -------------------------
with rhs:
    with rhs:
        # Build tile items from your existing data
        tile_items = build_tile_items(holdings, ref_map, prices)

        if tile_items:
            render_portfolio_tiles(tile_items)
        else:
            st.info("Tiles not available yet (need month reference + live price for active tickers).")

        st.markdown("---")
        with st.expander("Controls (upload, polling, triggers, email)", expanded=False):
            st.subheader("Reference prices")
            st.write("Use this if you want to recapture start-of-month references (no live backfill will be used).")
            if st.button("Reset this month's reference prices"):
                with SessionLocal() as session:
                    basket = ensure_basket(session)
                    mkey = month_key_now()
                    session.execute(
                        delete(MonthReference).where(
                            MonthReference.basket_id == basket.id,
                            MonthReference.month_key == mkey,
                        )
                    )
                    session.commit()
                st.success("Deleted this month's reference prices. They will be recreated on next refresh.")
                st.rerun()

            st.subheader("Upload")
            upload = st.file_uploader("Upload tickers (CSV or Excel) — max 24", type=["csv", "xlsx", "xls"])
            if upload is not None:
                new_tickers = parse_upload(upload)
                if len(new_tickers) == 0:
                    st.warning("No tickers found in the uploaded file.")
                else:
                    st.caption("Parsed tickers")
                    st.write(new_tickers)

                    with SessionLocal() as session:
                        basket = ensure_basket(session)
                        save_holdings_and_reference(session, basket, new_tickers)

                    st.success(
                        f"Loaded {len(new_tickers)} tickers and captured start-of-month reference prices (if not already set)."
                    )
                    st.rerun()

            st.subheader("Polling")
            poll_seconds = st.number_input(
                "Price refresh interval (seconds)",
                min_value=10,
                max_value=600,
                value=int(basket.poll_seconds),
                step=10,
            )
            if st.button("Save polling interval"):
                with SessionLocal() as session:
                    b = session.execute(select(Basket).where(Basket.id == basket.id)).scalar_one()
                    b.poll_seconds = int(poll_seconds)
                    session.commit()
                st.success("Saved.")
                st.rerun()

            st.subheader("Trigger levels")
            tp_levels = sorted([t.level_pct for t in triggers if t.kind == "TP"])
            sl_levels = sorted([t.level_pct for t in triggers if t.kind == "SL"])

            tp1 = st.number_input("Take Profit 1 (+)", value=float(tp_levels[0]) if len(tp_levels) > 0 else 0.0350,
                                  step=0.0001, format="%.4f")
            tp2 = st.number_input("Take Profit 2 (+)", value=float(tp_levels[1]) if len(tp_levels) > 1 else 0.0475,
                                  step=0.0001, format="%.4f")
            tp3 = st.number_input("Take Profit 3 (+)", value=float(tp_levels[2]) if len(tp_levels) > 2 else 0.0650,
                                  step=0.0001, format="%.4f")

            sl1 = st.number_input("Stop Loss 1 (-)", value=float(sl_levels[0]) if len(sl_levels) > 0 else -0.0600,
                                  step=0.0001, format="%.4f")
            sl2 = st.number_input("Stop Loss 2 (-)", value=float(sl_levels[1]) if len(sl_levels) > 1 else -0.0600,
                                  step=0.0001, format="%.4f")
            sl3 = st.number_input("Stop Loss 3 (-)", value=float(sl_levels[2]) if len(sl_levels) > 2 else -0.0600,
                                  step=0.0001, format="%.4f")

            if st.button("Save trigger levels"):
                with SessionLocal() as session:
                    session.execute(delete(TriggerLevel).where(TriggerLevel.basket_id == basket.id))
                    for lv in [tp1, tp2, tp3]:
                        session.add(TriggerLevel(basket_id=basket.id, kind="TP", level_pct=float(lv), enabled=True))
                    for lv in [sl1, sl2, sl3]:
                        session.add(TriggerLevel(basket_id=basket.id, kind="SL", level_pct=float(lv), enabled=True))
                    session.commit()
                st.success("Trigger levels updated.")
                st.rerun()

            st.subheader("Email recipients (SendGrid)")
            recipients_text = st.text_area("Recipients (one per line)",
                                           value=st.session_state.get("recipients_text", ""))
            st.session_state["recipients_text"] = recipients_text
