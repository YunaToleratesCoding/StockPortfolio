import datetime as dt
import time
import requests
import yfinance as yf
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

# =========================
# Page config
# =========================
st.set_page_config(page_title="Stock Insight Dashboard", layout="wide")

# =========================
# Secrets / Keys
# =========================
NEWS_API_KEY = st.secrets.get("NEWS_API_KEY", "pub_b94aa5942cf24b209ce26666d30b5207")

# =========================
# Header
# =========================
st.markdown(
    """
    <h1 style="margin-bottom:0">Stock Insight Dashboard</h1>
    <p style="color:#555;font-size:16px;margin-top:6px">
        Compare performance, explore correlations, view insider activity, and scan recent news.
        Built with Python + Streamlit + yfinance.
    </p>
    """,
    unsafe_allow_html=True,
)
st.markdown("---")

# =========================
# Inputs (time range under compare ticker)
# =========================
left, right = st.columns([2, 1])

with left:
    symbol = st.text_input("Primary stock symbol", "AAPL").upper().strip()
    compare_symbol = st.text_input("Compare with another symbol (optional)", "").upper().strip()
    range_option = st.selectbox(
        "Time range",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd", "max"],
        index=2,
    )
    fetch_btn = st.button("Fetch / Refresh Data", type="primary")

with right:
    st.subheader("Display options")
    show_ma = st.checkbox("Show moving averages (20/50)", value=True)
    chart_type = st.radio("Chart type", ["Line", "Candlestick"], index=0)
    normalize_compare = st.toggle(
        "Normalize to % change when comparing",
        value=True,
        help="Recommended when prices differ a lot (e.g., NVDA vs AAPL)."
    )
    with st.expander("Advanced options"):
        use_log = st.checkbox("Use logarithmic scale (advanced)", value=False)

st.markdown(
    """
    **Tip:** On a public deployment, data providers may rate-limit.  
    This app caches results and only fetches new prices when you click **Fetch / Refresh Data**.
    """
)

# =========================
# Data helpers
# =========================
def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df

@st.cache_data(ttl=3600, show_spinner=False, max_entries=256)  # cache 1 hour
def fetch_stock_data_cached(sym: str, period: str) -> pd.DataFrame:
    """
    Cached Yahoo fetch. Kept intentionally simple/robust.
    If Yahoo throttles, callers handle empty df.
    """
    if not sym:
        return pd.DataFrame()

    # Retry lightly (helps with transient issues)
    for attempt in range(2):
        try:
            df = yf.download(sym, period=period, auto_adjust=False, progress=False, threads=False)
            df = _flatten_cols(df).dropna()
            return df
        except Exception:
            # brief backoff
            time.sleep(0.7 * (attempt + 1))
            continue

    return pd.DataFrame()

def add_mas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Close" in out.columns and len(out) >= 20:
        out["MA20"] = out["Close"].rolling(20).mean()
    if "Close" in out.columns and len(out) >= 50:
        out["MA50"] = out["Close"].rolling(50).mean()
    return out

def compute_metrics(df: pd.DataFrame) -> dict:
    if df is None or df.empty or "Close" not in df.columns or len(df) < 2:
        return {}
    close = df["Close"].dropna()
    if len(close) < 2:
        return {}

    last = float(close.iloc[-1])
    prev = float(close.iloc[-2])
    day_pct = (last / prev - 1) * 100 if prev else 0.0

    first = float(close.iloc[0])
    period_pct = (last / first - 1) * 100 if first else 0.0

    rets = close.pct_change().dropna()
    vol = float(rets.std() * 100) if len(rets) > 5 else float("nan")

    return {
        "Last Close": last,
        "Day %": day_pct,
        "Period %": period_pct,
        "Vol (daily std %)": vol,
        "Last Date": df.index[-1].date(),
    }

def plot_line(df: pd.DataFrame, title: str, y_cols: list[str], y_title: str = "") -> go.Figure:
    fig = go.Figure()
    for c in y_cols:
        if c in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[c], mode="lines", name=c))
    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=55, b=10),
        height=420,
    )
    fig.update_yaxes(title_text=y_title)
    if use_log:
        fig.update_yaxes(type="log")
    return fig

def plot_candles(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="OHLC"
    ))
    if show_ma:
        if "MA20" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], mode="lines", name="MA20"))
        if "MA50" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], mode="lines", name="MA50"))
    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=55, b=10),
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    if use_log:
        fig.update_yaxes(type="log")
    return fig

# =========================
# Session state: store last fetched data
# =========================
if "df1" not in st.session_state:
    st.session_state.df1 = pd.DataFrame()
if "df2" not in st.session_state:
    st.session_state.df2 = pd.DataFrame()
if "last_fetch" not in st.session_state:
    st.session_state.last_fetch = None

# Only fetch when user clicks (prevents constant reruns from spamming Yahoo)
if fetch_btn:
    with st.spinner("Fetching market data..."):
        st.session_state.df1 = fetch_stock_data_cached(symbol, range_option)
        st.session_state.df2 = fetch_stock_data_cached(compare_symbol, range_option) if compare_symbol else pd.DataFrame()
        st.session_state.last_fetch = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

df1 = st.session_state.df1
df2 = st.session_state.df2

# Add MAs after fetch
if not df1.empty and show_ma:
    df1 = add_mas(df1)
if not df2.empty and show_ma:
    df2 = add_mas(df2)

# Show last fetch time
if st.session_state.last_fetch:
    st.caption(f"Last refresh: {st.session_state.last_fetch}")

# =========================
# Price Analysis
# =========================
st.markdown("## Price Analysis")

if df1.empty:
    st.warning(
        "No data loaded yet (or Yahoo throttled the request). "
        "Click **Fetch / Refresh Data**. If it still fails, wait a minute and try again."
    )
else:
    m = compute_metrics(df1)
    if m:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Last Close", f"${m['Last Close']:.2f}")
        c2.metric("Day Change", f"{m['Day %']:.2f}%")
        c3.metric("Period Change", f"{m['Period %']:.2f}%")
        c4.metric("Daily Vol (std)", f"{m['Vol (daily std %)']:.2f}%" if m["Vol (daily std %)"] == m["Vol (daily std %)"] else "n/a")

    if compare_symbol and not df2.empty:
        st.subheader(f"{symbol} vs {compare_symbol}")

        combined = pd.concat(
            [df1["Close"].rename(symbol), df2["Close"].rename(compare_symbol)],
            axis=1
        ).dropna()

        if combined.empty:
            st.info("Not enough overlapping dates to compare.")
        else:
            if normalize_compare:
                norm = (combined / combined.iloc[0] - 1) * 100
                st.plotly_chart(
                    plot_line(norm, "Comparison (Normalized % Change)", [symbol, compare_symbol], y_title="Percent change (%)"),
                    width="stretch"
                )
            else:
                st.plotly_chart(
                    plot_line(combined, "Comparison (Raw Close Price)", [symbol, compare_symbol], y_title="Price"),
                    width="stretch"
                )

            with st.expander("Show each ticker’s own chart"):
                st.plotly_chart(
                    plot_line(df1, f"{symbol} Close", ["Close"] + (["MA20", "MA50"] if show_ma else []), y_title="Price"),
                    width="stretch"
                )
                st.plotly_chart(
                    plot_line(df2, f"{compare_symbol} Close", ["Close"] + (["MA20", "MA50"] if show_ma else []), y_title="Price"),
                    width="stretch"
                )
    else:
        # Single stock view
        if chart_type == "Candlestick":
            needed = {"Open", "High", "Low", "Close"}
            if needed.issubset(df1.columns):
                st.plotly_chart(plot_candles(df1, f"{symbol} Candlestick"), width="stretch")
            else:
                st.info("Candlestick view requires OHLC columns. Showing line chart instead.")
                st.plotly_chart(
                    plot_line(df1, f"{symbol} Close", ["Close"] + (["MA20", "MA50"] if show_ma else []), y_title="Price"),
                    width="stretch"
                )
        else:
            st.plotly_chart(
                plot_line(df1, f"{symbol} Close", ["Close"] + (["MA20", "MA50"] if show_ma else []), y_title="Price"),
                width="stretch"
            )

        with st.expander("Show OHLC lines"):
            ohlc_cols = [c for c in ["Open", "High", "Low", "Close"] if c in df1.columns]
            if ohlc_cols:
                st.plotly_chart(plot_line(df1, f"{symbol} OHLC", ohlc_cols, y_title="Price"), width="stretch")
            else:
                st.info("OHLC columns not available.")

# =========================
# Correlation (on-demand to avoid rate limiting)
# =========================
st.markdown("## Cross-Stock Correlation")

corr_input = st.text_input("Symbols (comma separated)", "AAPL, MSFT, GOOGL")
corr_syms = [s.strip().upper() for s in corr_input.split(",") if s.strip()]

run_corr = st.button("Run correlation (fetches data)", help="This triggers multiple downloads; keep it on-demand to avoid throttling.")

if run_corr:
    with st.spinner("Fetching symbols for correlation..."):
        corr_data = {}
        for s in corr_syms[:10]:  # cap to avoid hammering Yahoo
            tmp = fetch_stock_data_cached(s, range_option)
            if not tmp.empty and "Close" in tmp.columns:
                corr_data[s] = tmp["Close"]

    if corr_data:
        corr_df = pd.DataFrame(corr_data).dropna()
        if corr_df.empty:
            st.info("Not enough overlapping data to compute correlations.")
        else:
            fig, ax = plt.subplots()
            sns.heatmap(corr_df.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
    else:
        st.warning("Could not fetch correlation data (rate limit or invalid symbols). Try again later.")

# =========================
# Analyst Summary
# =========================
st.markdown("## Analyst Summary")

def generate_summary(df: pd.DataFrame, sym: str, period_label: str) -> str:
    if df.empty or "Close" not in df.columns or len(df) < 2:
        return "Not enough data to generate a summary."

    close = df["Close"].dropna()
    first = float(close.iloc[0])
    last = float(close.iloc[-1])
    pct = (last / first - 1) * 100 if first else 0.0

    daily_rets = close.pct_change().dropna()
    vol = float(daily_rets.std() * 100) if len(daily_rets) > 5 else None

    text = f"{sym} moved {pct:.2f}% over the selected period ({period_label}). "
    if pct > 5:
        text += "Trend appears positive; consider checking catalysts (earnings, sector strength, headlines). "
    elif pct < -5:
        text += "Trend appears negative; check recent news, guidance changes, or broad market moves. "
    else:
        text += "Price action has been relatively stable. "

    if vol is not None:
        text += f"Observed daily volatility (std of returns) is about {vol:.2f}%."

    return text

if not df1.empty:
    st.info(generate_summary(df1, symbol, range_option))
else:
    st.info("Load price data to generate a summary.")

# =========================
# Insider Trading Snapshot (Finviz scrape) - cached and safe
# =========================
st.markdown("## Insider Trading Snapshot")

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_insider_finviz(sym: str) -> pd.DataFrame:
    if not sym:
        return pd.DataFrame()
    try:
        url = f"https://finviz.com/quote.ashx?t={sym}"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        table = soup.find("table", class_="body-table")
        rows = table.find_all("tr") if table else []

        data = []
        for row in rows:
            cols = row.find_all("td")
            if len(cols) == 7:
                data.append([c.get_text(strip=True) for c in cols])

        if not data:
            return pd.DataFrame()

        return pd.DataFrame(
            data,
            columns=["Insider", "Relationship", "Date", "Transaction", "Cost", "Shares", "Value"],
        )
    except Exception:
        return pd.DataFrame()

with st.expander("Load insider data (cached)"):
    insider_df = fetch_insider_finviz(symbol)
    if insider_df.empty:
        st.info("No insider data found (or the source blocked the request).")
    else:
        st.dataframe(insider_df, width="stretch")

# =========================
# Event-Driven Insights (NewsData.io) - cached
# =========================
st.markdown("## Event-Driven Insights")

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_news(sym: str):
    if not NEWS_API_KEY or not sym:
        return []
    url = f"https://newsdata.io/api/1/news?apikey={NEWS_API_KEY}&q={sym}&language=en"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        return data.get("results", [])[:6] if isinstance(data, dict) else []
    except Exception:
        return []

with st.expander("Load recent news (cached)"):
    news_items = fetch_news(symbol)
    if news_items:
        for item in news_items:
            title = item.get("title", "(no title)")
            link = item.get("link", "")
            pub = item.get("pubDate", "")
            st.markdown(f"**{title}**")
            st.markdown(f"{pub} — [Link]({link})" if link else f"{pub}")
            st.write("")
    else:
        st.info("No recent news found (or API is rate-limited).")

# =========================
# Footer
# =========================
st.markdown("---")
st.markdown(
    "<p style='font-size:13px;color:#777'>Portfolio project: Python, Streamlit, yfinance, Plotly, seaborn, requests, BeautifulSoup.</p>",
    unsafe_allow_html=True,
)
