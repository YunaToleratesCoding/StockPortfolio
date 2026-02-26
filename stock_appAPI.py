import time
import datetime as dt
import requests
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------
# Page config + minimal styling
# -----------------------------
st.set_page_config(page_title="Stock Insight Dashboard", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.1rem; padding-bottom: 2rem; }
      .muted { color: rgba(49,51,63,0.70); font-size: 0.95rem; }
      .card { border: 1px solid rgba(49,51,63,0.12); border-radius: 14px; padding: 12px 14px; background: white; }
      hr { margin: 1.0rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

NEWS_API_KEY = st.secrets.get("NEWS_API_KEY", "")

st.title("Stock Insight Dashboard")
st.markdown('<div class="muted">Price history, comparisons, correlation, narrative summaries, insider/institutional, news, and government trading.</div>', unsafe_allow_html=True)
st.markdown("---")


# -----------------------------
# Period helpers
# -----------------------------
def period_to_start(period: str) -> dt.date | None:
    today = dt.date.today()
    if period == "1mo":
        return today - dt.timedelta(days=35)
    if period == "3mo":
        return today - dt.timedelta(days=100)
    if period == "6mo":
        return today - dt.timedelta(days=210)
    if period == "1y":
        return today - dt.timedelta(days=370)
    if period == "2y":
        return today - dt.timedelta(days=740)
    if period == "5y":
        return today - dt.timedelta(days=5 * 370)
    if period == "ytd":
        return dt.date(today.year, 1, 1)
    if period == "max":
        return None
    return None


def flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.dropna()
    return df


def normalize_pct_change(series: pd.Series) -> pd.Series:
    s = series.dropna()
    if s.empty:
        return s
    base = s.iloc[0]
    if pd.isna(base) or base == 0:
        return s * 0
    return (s / base - 1) * 100


# -----------------------------
# Fetchers (Yahoo primary, Stooq fallback)
# -----------------------------
@st.cache_data(ttl=3600, show_spinner=False, max_entries=256)
def fetch_from_yahoo(symbol: str, period: str) -> pd.DataFrame:
    # Lower request intensity to reduce throttling risk
    df = yf.download(
        symbol,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    return flatten_yf_columns(df)


@st.cache_data(ttl=3600, show_spinner=False, max_entries=256)
def fetch_from_stooq(symbol: str, period: str) -> pd.DataFrame:
    """
    Free fallback provider (no key).
    For US tickers, Stooq often expects ".us" (e.g., aapl.us).
    Returns columns: Open, High, Low, Close, Volume with a DatetimeIndex.
    """
    sym = symbol.lower()
    # Try a few common variants
    candidates = [sym, f"{sym}.us", f"{sym}.us"] if not sym.endswith(".us") else [sym]

    start = period_to_start(period)

    for c in candidates:
        url = f"https://stooq.com/q/d/l/?s={c}&i=d"
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            df = pd.read_csv(pd.compat.StringIO(r.text)) if hasattr(pd.compat, "StringIO") else pd.read_csv(url)
        except Exception:
            # If pd.compat.StringIO isn't available, retry using pandas read_csv(url)
            try:
                df = pd.read_csv(url)
            except Exception:
                continue

        if df is None or df.empty or "Date" not in df.columns:
            continue

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date").sort_index()

        # Standardize column names to match Yahoo
        rename_map = {c: c.title() for c in df.columns}
        df = df.rename(columns=rename_map)

        # Keep only expected columns
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        df = df[keep].dropna()

        if start is not None:
            df = df[df.index.date >= start]

        if not df.empty:
            return df

    return pd.DataFrame()


def fetch_price_data(symbol: str, period: str) -> tuple[pd.DataFrame, str]:
    """
    Returns (df, source_name).
    """
    if not symbol:
        return pd.DataFrame(), "none"

    # Try Yahoo first with light retry/backoff
    for attempt in range(2):
        try:
            df = fetch_from_yahoo(symbol, period)
            if not df.empty:
                return df, "yahoo"
        except Exception:
            time.sleep(0.6 * (attempt + 1))

    # Fallback to Stooq
    df2 = fetch_from_stooq(symbol, period)
    if not df2.empty:
        return df2, "stooq"

    return pd.DataFrame(), "none"


# -----------------------------
# Inputs (time range under compare)
# -----------------------------
colA, colB = st.columns([1.1, 1.1], gap="large")
with colA:
    symbol = st.text_input("Primary stock", "AAPL").strip().upper()
with colB:
    compare_symbol = st.text_input("Optional second stock (comparison)", "").strip().upper()

range_option = st.selectbox("Time range", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd", "max"], index=2)

opt_left, opt_right = st.columns([1.4, 1.0], gap="large")
with opt_left:
    fetch_btn = st.button("Fetch / Refresh Data", type="primary")
with opt_right:
    show_pct_compare = st.checkbox("Compare as percent change (recommended)", value=True, disabled=not bool(compare_symbol))
    with st.expander("Advanced options"):
        use_log_scale = st.checkbox("Use log scale (raw comparison only)", value=False, disabled=not bool(compare_symbol))
        show_volume = st.checkbox("Show volume", value=False)

st.caption("This app fetches only when you click Fetch/Refresh to avoid rate limits on public deployments.")

# -----------------------------
# Session state for fetched data (prevents auto refetch)
# -----------------------------
if "df1" not in st.session_state:
    st.session_state.df1 = pd.DataFrame()
    st.session_state.src1 = "none"
if "df2" not in st.session_state:
    st.session_state.df2 = pd.DataFrame()
    st.session_state.src2 = "none"
if "last_fetch" not in st.session_state:
    st.session_state.last_fetch = None

if fetch_btn:
    with st.spinner("Fetching data..."):
        df1, src1 = fetch_price_data(symbol, range_option)
        df2, src2 = (pd.DataFrame(), "none")
        if compare_symbol:
            df2, src2 = fetch_price_data(compare_symbol, range_option)

        st.session_state.df1 = df1
        st.session_state.src1 = src1
        st.session_state.df2 = df2
        st.session_state.src2 = src2
        st.session_state.last_fetch = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

df1 = st.session_state.df1
df2 = st.session_state.df2

if st.session_state.last_fetch:
    st.caption(f"Last refresh: {st.session_state.last_fetch}")

# -----------------------------
# Price charts
# -----------------------------
st.subheader("Price charts")

if df1.empty:
    st.warning(
        "No price data loaded yet (or both Yahoo and fallback source failed). "
        "Click Fetch/Refresh. If it still fails, try a different ticker."
    )
else:
    st.info(f"Primary data source: {st.session_state.src1.upper()}")

    if not compare_symbol:
        st.markdown(f"<div class='card'><b>{symbol}</b> — Close</div>", unsafe_allow_html=True)

        fig = px.line(df1, x=df1.index, y="Close", title=None)
        fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=10, b=10), height=420)
        st.plotly_chart(fig, width="stretch")

        if show_volume and "Volume" in df1.columns:
            vfig = px.bar(df1, x=df1.index, y="Volume", title=None)
            vfig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=10, b=10), height=260)
            st.plotly_chart(vfig, width="stretch")

        # Candlestick
        if set(["Open", "High", "Low", "Close"]).issubset(df1.columns):
            st.markdown("<div class='card'><b>OHLC</b> (candlestick)</div>", unsafe_allow_html=True)
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
            cfig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=10, b=10), height=520)
            st.plotly_chart(cfig, width="stretch")
        else:
            st.caption("OHLC not available from this source for this symbol.")
    else:
        if df2.empty:
            st.warning("Comparison ticker did not load. Try Fetch/Refresh again or choose another ticker.")
        else:
            st.info(f"Comparison data source: {st.session_state.src2.upper()}")

            s1 = df1["Close"].rename(symbol)
            s2 = df2["Close"].rename(compare_symbol)
            combined = pd.concat([s1, s2], axis=1).dropna()

            if combined.empty:
                st.warning("Not enough overlapping dates between the two tickers in this time range.")
            else:
                st.markdown(f"<div class='card'><b>{symbol}</b> vs <b>{compare_symbol}</b></div>", unsafe_allow_html=True)

                if show_pct_compare:
                    pct = combined.apply(normalize_pct_change, axis=0)
                    pfig = px.line(pct, x=pct.index, y=pct.columns, title=None)
                    pfig.update_layout(template="plotly_white", yaxis_title="Percent change (%)",
                                       margin=dict(l=10, r=10, t=10, b=10), height=420)
                    st.plotly_chart(pfig, width="stretch")
                else:
                    rfig = px.line(combined, x=combined.index, y=combined.columns, title=None)
                    if use_log_scale:
                        rfig.update_yaxes(type="log")
                    rfig.update_layout(template="plotly_white", yaxis_title="Price",
                                       margin=dict(l=10, r=10, t=10, b=10), height=420)
                    st.plotly_chart(rfig, width="stretch")

# -----------------------------
# Correlation (on-demand to avoid rate limiting)
# -----------------------------
st.subheader("Cross-stock correlation explorer")
corr_input = st.text_input("Symbols (comma-separated)", "AAPL,MSFT,GOOGL").strip()
corr_period = st.selectbox("Correlation time range", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
run_corr = st.button("Run correlation (fetches data)")

if run_corr:
    syms = [s.strip().upper() for s in corr_input.split(",") if s.strip()][:10]
    closes = {}
    with st.spinner("Fetching symbols for correlation..."):
        for s in syms:
            dfx, _src = fetch_price_data(s, corr_period)
            if not dfx.empty and "Close" in dfx.columns:
                closes[s] = dfx["Close"]

    if len(closes) < 2:
        st.warning("Could not load enough symbols for correlation (provider may be rate-limiting).")
    else:
        corr_df = pd.DataFrame(closes).dropna()
        corr = corr_df.pct_change().dropna().corr()
        heat = px.imshow(corr, text_auto=True, aspect="auto", title=None)
        heat.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=10, b=10), height=500)
        st.plotly_chart(heat, width="stretch")

# -----------------------------
# Analyst summary
# -----------------------------
st.subheader("Analyst-style summary")

def generate_narrative(df: pd.DataFrame, sym: str) -> str:
    if df is None or df.empty or "Close" not in df.columns or len(df) < 5:
        return "Not enough data to generate a summary for this time range."

    close = df["Close"].dropna()
    if close.empty:
        return "Not enough closing prices to generate a summary."

    start = float(close.iloc[0])
    end = float(close.iloc[-1])
    pct = ((end / start) - 1.0) * 100.0 if start else 0.0

    daily_returns = close.pct_change().dropna()
    vol = (daily_returns.std() * (252 ** 0.5) * 100.0) if len(daily_returns) > 10 else None

    roll_max = close.cummax()
    dd = (close / roll_max - 1.0)
    mdd = (dd.min() * 100.0) if not dd.empty else None

    direction = "up" if pct >= 0 else "down"
    tone = "strong" if abs(pct) >= 10 else "moderate" if abs(pct) >= 3 else "mild"

    parts = [
        f"{sym} is {direction} {abs(pct):.2f}% over the selected period ({close.index[0].date()} to {close.index[-1].date()}).",
        f"The move is {tone} for this window."
    ]
    if vol is not None:
        parts.append(f"Estimated annualized volatility is ~{vol:.1f}%.")
    if mdd is not None:
        parts.append(f"Maximum drawdown during the window was {mdd:.1f}%.")

    return " ".join(parts)

if not df1.empty:
    st.info(generate_narrative(df1, symbol))
else:
    st.info("Load price data to generate a narrative summary.")

# -----------------------------
# Government trades (Capitol Trades) – optional section stays separate
# -----------------------------
st.subheader("Government trading activity (Capitol Trades)")
st.caption("This section can be added next; it does not depend on Yahoo. If you want it live now, say so and I’ll merge it back in cleanly.")

# -----------------------------
# Debug
# -----------------------------
with st.expander("Debug: view raw data"):
    st.write("Primary data")
    st.dataframe(df1.tail(20), width="stretch")
    if compare_symbol:
        st.write("Comparison data")
        st.dataframe(df2.tail(20), width="stretch")
