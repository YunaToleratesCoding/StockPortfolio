import time
import requests
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------
# Page + Style
# -----------------------------
st.set_page_config(page_title="Stock Insight Dashboard", layout="wide")

CUSTOM_CSS = """
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
h1, h2, h3 {letter-spacing: -0.2px;}
.small-muted {color: rgba(49,51,63,0.65); font-size: 0.92rem;}
.card {
  border: 1px solid rgba(49,51,63,0.12);
  border-radius: 14px;
  padding: 14px 14px;
  background: white;
}
hr {margin: 1.0rem 0;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.title("Stock Insight Dashboard")
st.markdown('<div class="small-muted">Price history, comparisons, correlation, narrative summaries, and government trading activity.</div>', unsafe_allow_html=True)
st.write("")

# -----------------------------
# Helpers
# -----------------------------
def _flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance sometimes returns MultiIndex columns: ('Close','AAPL'). Flatten to 'Close'."""
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df

def _safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def _normalize_to_pct_change(series: pd.Series) -> pd.Series:
    series = series.dropna()
    if series.empty:
        return series
    base = series.iloc[0]
    if base == 0 or pd.isna(base):
        return series * 0
    return (series / base - 1.0) * 100.0

# -----------------------------
# Data fetching (cached)
# -----------------------------
@st.cache_data(ttl=900, show_spinner=False)  # 15 minutes
def fetch_stock_data(symbol: str, period: str) -> pd.DataFrame:
    """
    Cached yfinance fetch. Caching is critical on Streamlit Cloud to avoid rate limits.
    """
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return pd.DataFrame()

    # Reduce yfinance load
    df = yf.download(symbol, period=period, interval="1d", auto_adjust=False, progress=False, threads=False)
    df = _flatten_yf_columns(df)
    if df is None or df.empty:
        return pd.DataFrame()

    # ensure index is datetime
    df.index = pd.to_datetime(df.index)
    return df

# CapitolTrades BFF endpoint (no key)
CAPITOL_BFF_TRADES = "https://bff.capitoltrades.com/trades"

@st.cache_data(ttl=1800, show_spinner=False)  # 30 minutes
def fetch_capitol_trades_pages(pages: int = 2, page_size: int = 100) -> pd.DataFrame:
    """
    Pulls recent trades from CapitolTrades internal API.
    We fetch a couple pages and filter client-side.
    Endpoint discussed publicly: bff.capitoltrades.com/trades (see StackOverflow examples).
    """
    all_rows = []
    for page in range(1, pages + 1):
        params = {
            "page": page,
            "pageSize": page_size,
            "per_page": page_size,
        }
        r = requests.get(CAPITOL_BFF_TRADES, params=params, timeout=20)
        r.raise_for_status()
        payload = r.json()

        data = payload.get("data", [])
        for row in data:
            pol = row.get("politician", {}) or {}
            asset = row.get("asset", {}) or {}

            all_rows.append({
                "published": row.get("published"),
                "traded": row.get("traded"),
                "filed_after_days": row.get("filedAfter"),
                "transaction": row.get("type"),
                "size": row.get("size"),
                "ticker": asset.get("assetTicker"),
                "asset_name": asset.get("assetName"),
                "politician": f"{pol.get('firstName','')} {pol.get('lastName','')}".strip(),
                "party": pol.get("party"),
                "chamber": pol.get("chamber"),
                "state": pol.get("_stateId"),
                "source_url": row.get("sourceUrl") or row.get("url") or None,
            })

        # small pause to be polite
        time.sleep(0.2)

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df

    # parse dates if present
    for col in ["published", "traded"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # clean tickers like "AAPL:US" -> "AAPL"
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str).str.replace(":US", "", regex=False)

    # newest first
    df = df.sort_values(by=["published", "traded"], ascending=False)
    return df

# -----------------------------
# Inputs (layout requirement: time range under compare)
# -----------------------------
colA, colB = st.columns([1.1, 1.1], gap="large")

with colA:
    symbol = st.text_input("Primary stock", "AAPL").strip().upper()

with colB:
    compare_symbol = st.text_input("Optional second stock (comparison)", "").strip().upper()

range_option = st.selectbox(
    "Time range",
    ["1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd", "max"],
    index=2
)

st.write("")

# -----------------------------
# Fetch + Validate
# -----------------------------
df1 = fetch_stock_data(symbol, range_option) if symbol else pd.DataFrame()
df2 = fetch_stock_data(compare_symbol, range_option) if compare_symbol else pd.DataFrame()

# yfinance rate limit handling message
if symbol and df1.empty:
    st.error("Primary stock data did not load. This can happen if yfinance is rate-limiting. Try again in a minute, or change the ticker.")
    st.stop()

# -----------------------------
# Price Charts
# -----------------------------
st.subheader("Price charts")

chart_col1, chart_col2 = st.columns([1.4, 1.0], gap="large")

with chart_col2:
    show_pct_compare = st.checkbox("Compare as percent change (recommended)", value=True, disabled=not bool(compare_symbol))
    with st.expander("Advanced display options"):
        use_log_scale = st.checkbox("Use log scale (comparison only)", value=False, disabled=not bool(compare_symbol))
        show_volume = st.checkbox("Show volume bars", value=False)

with chart_col1:
    if not compare_symbol:
        # Single stock view
        st.markdown(f"<div class='card'><b>{symbol}</b> — closing price</div>", unsafe_allow_html=True)

        fig = px.line(df1, x=df1.index, y="Close", title=None)
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")

        if show_volume and "Volume" in df1.columns:
            vfig = px.bar(df1, x=df1.index, y="Volume", title=None)
            vfig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(vfig, width="stretch")

        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown("<div class='card'><b>OHLC</b> — single stock</div>", unsafe_allow_html=True)

        # Candlestick is more “finance-professional” than 4 lines
        cfig = go.Figure(
            data=[go.Candlestick(
                x=df1.index,
                open=df1["Open"],
                high=df1["High"],
                low=df1["Low"],
                close=df1["Close"],
                name=symbol
            )]
        )
        cfig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(cfig, width="stretch")

    else:
        # Comparison view
        if df2.empty:
            st.error("Second stock data did not load. Try again later or change the ticker.")
        else:
            st.markdown(f"<div class='card'><b>{symbol}</b> vs <b>{compare_symbol}</b></div>", unsafe_allow_html=True)

            s1 = df1["Close"].rename(symbol)
            s2 = df2["Close"].rename(compare_symbol)
            combined = pd.concat([s1, s2], axis=1).dropna()

            if combined.empty:
                st.error("No overlapping dates between the two symbols in this time range.")
            else:
                if show_pct_compare:
                    pct = combined.apply(_normalize_to_pct_change, axis=0)
                    pfig = px.line(pct, x=pct.index, y=pct.columns, title=None)
                    pfig.update_layout(
                        yaxis_title="Percent change (%)",
                        margin=dict(l=10, r=10, t=10, b=10)
                    )
                    st.plotly_chart(pfig, width="stretch")
                else:
                    rfig = px.line(combined, x=combined.index, y=combined.columns, title=None)
                    if use_log_scale:
                        rfig.update_yaxes(type="log")
                    rfig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
                    st.plotly_chart(rfig, width="stretch")

                if show_volume:
                    # show separate volume bars in a compact way
                    vol1 = df1["Volume"].rename(symbol) if "Volume" in df1.columns else None
                    vol2 = df2["Volume"].rename(compare_symbol) if "Volume" in df2.columns else None
                    if vol1 is not None and vol2 is not None:
                        v = pd.concat([vol1, vol2], axis=1).dropna()
                        vfig = px.line(v, x=v.index, y=v.columns, title=None)
                        vfig.update_layout(yaxis_title="Volume", margin=dict(l=10, r=10, t=10, b=10))
                        st.plotly_chart(vfig, width="stretch")

# -----------------------------
# Analyst-style summary
# -----------------------------
st.subheader("Analyst-style summary")

def generate_narrative(df: pd.DataFrame, sym: str) -> str:
    if df is None or df.empty or "Close" not in df.columns or len(df) < 5:
        return "Not enough data to generate a summary for this time range."

    close = df["Close"].dropna()
    if close.empty:
        return "Not enough valid closing prices to generate a summary."

    start = close.iloc[0]
    end = close.iloc[-1]
    pct = ((end / start) - 1.0) * 100.0 if start else 0.0

    daily_returns = close.pct_change().dropna()
    vol = daily_returns.std() * (252 ** 0.5) * 100.0 if not daily_returns.empty else None

    # max drawdown
    roll_max = close.cummax()
    dd = (close / roll_max - 1.0)
    mdd = dd.min() * 100.0 if not dd.empty else None

    direction = "up" if pct >= 0 else "down"
    tone = "strong" if abs(pct) >= 10 else "moderate" if abs(pct) >= 3 else "mild"

    parts = []
    parts.append(f"{sym} is {direction} {abs(pct):.2f}% over the selected period ({close.index[0].date()} to {close.index[-1].date()}).")
    parts.append(f"The move is {tone} relative to the chosen window.")
    if vol is not None:
        parts.append(f"Estimated annualized volatility over the window is ~{vol:.1f}%.")
    if mdd is not None:
        parts.append(f"Maximum drawdown during the window was {mdd:.1f}%.")

    return " ".join(parts)

st.info(generate_narrative(df1, symbol))

# -----------------------------
# Correlation Explorer
# -----------------------------
st.subheader("Cross-stock correlation explorer")

corr_input = st.text_input("Symbols (comma-separated)", "AAPL,MSFT,GOOGL").strip()
corr_list = [s.strip().upper() for s in corr_input.split(",") if s.strip()]
corr_period = st.selectbox("Correlation time range", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)

if len(corr_list) < 2:
    st.warning("Enter at least two symbols for correlation.")
else:
    closes = {}
    for sym in corr_list[:10]:  # hard cap to avoid rate limits
        dfx = fetch_stock_data(sym, corr_period)
        if not dfx.empty and "Close" in dfx.columns:
            closes[sym] = dfx["Close"]

    if len(closes) < 2:
        st.warning("Could not load enough symbols to compute correlation (yfinance may be rate-limiting).")
    else:
        corr_df = pd.DataFrame(closes).dropna()
        corr = corr_df.pct_change().dropna().corr()

        heat = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            title=None
        )
        heat.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(heat, width="stretch")

# -----------------------------
# Insider / Institutional (yfinance-based)
# -----------------------------
st.subheader("Insider and institutional activity")

ticker_obj = yf.Ticker(symbol)

col_i1, col_i2 = st.columns([1, 1], gap="large")

with col_i1:
    st.markdown("<div class='card'><b>Major holders</b></div>", unsafe_allow_html=True)
    try:
        mh = ticker_obj.major_holders
        if mh is not None and not mh.empty:
            st.dataframe(mh, width="stretch")
        else:
            st.write("No major holders data available for this ticker.")
    except Exception:
        st.write("Major holders data unavailable.")

with col_i2:
    st.markdown("<div class='card'><b>Institutional holders</b></div>", unsafe_allow_html=True)
    try:
        ih = ticker_obj.institutional_holders
        if ih is not None and not ih.empty:
            st.dataframe(ih.head(15), width="stretch")
        else:
            st.write("No institutional holders data available for this ticker.")
    except Exception:
        st.write("Institutional holders data unavailable.")

# -----------------------------
# Government trading (CapitolTrades)
# -----------------------------
st.subheader("Government trading activity (Capitol Trades)")

tracked = [
    "Pelosi",
    "McConnell",
    "Tuberville",
    "Gottheimer",
    "Khanna",
    "Greene",
    "Larsen",
    "DelBene",
]
trader_mode = st.radio("Mode", ["Tracked trader", "Search any name"], horizontal=True)

name_filter = ""
if trader_mode == "Tracked trader":
    name_filter = st.selectbox("Select a trader", tracked, index=0)
else:
    name_filter = st.text_input("Enter a name to filter (e.g., Pelosi)", "Pelosi").strip()

gov_col1, gov_col2 = st.columns([1, 1], gap="large")
with gov_col1:
    pages_to_pull = st.slider("How much recent data to pull (pages)", min_value=1, max_value=5, value=2)
with gov_col2:
    show_rows = st.slider("Rows to show", min_value=5, max_value=50, value=15)

try:
    trades_df = fetch_capitol_trades_pages(pages=pages_to_pull, page_size=100)

    if trades_df.empty:
        st.info("No government trades returned from the source right now.")
    else:
        if name_filter:
            mask = trades_df["politician"].str.contains(name_filter, case=False, na=False)
            filtered = trades_df.loc[mask].copy()
        else:
            filtered = trades_df.copy()

        if filtered.empty:
            st.info("No trades matched that name in the recent pages pulled.")
        else:
            # nicer columns
            out = filtered.head(show_rows)[
                ["published", "politician", "party", "chamber", "state", "ticker", "asset_name", "transaction", "size", "filed_after_days"]
            ].copy()

            out.rename(columns={
                "published": "Published",
                "politician": "Politician",
                "party": "Party",
                "chamber": "Chamber",
                "state": "State",
                "ticker": "Ticker",
                "asset_name": "Asset",
                "transaction": "Transaction",
                "size": "Size",
                "filed_after_days": "Filed After (days)"
            }, inplace=True)

            st.dataframe(out, width="stretch")

            # quick “top tickers” summary
            top = filtered["ticker"].dropna()
            top = top[top != "None"]
            if not top.empty:
                top_counts = top.value_counts().head(8).reset_index()
                top_counts.columns = ["Ticker", "Mentions"]
                bar = px.bar(top_counts, x="Ticker", y="Mentions", title=None)
                bar.update_layout(margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(bar, width="stretch")

except Exception as e:
    st.warning(f"Government trades temporarily unavailable: {e}")

# -----------------------------
# Raw data (debug)
# -----------------------------
with st.expander("Debug: view raw price data"):
    st.write(f"Primary: {symbol} ({range_option})")
    st.dataframe(df1.tail(20), width="stretch")
    if compare_symbol:
        st.write(f"Compare: {compare_symbol} ({range_option})")
        st.dataframe(df2.tail(20), width="stretch")
