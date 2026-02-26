import os
import datetime as dt
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
# Keys (put these in Streamlit secrets for public apps)
# =========================
# NewsData.io key (you already use this)
NEWS_API_KEY = st.secrets.get("NEWS_API_KEY", "pub_b94aa5942cf24b209ce26666d30b5207")

# Optional: Quiver Quant API key for Congress trades
# Quiver requires an API token to access datasets like Congress Trading. :contentReference[oaicite:1]{index=1}
QUIVER_API_KEY = st.secrets.get("QUIVER_API_KEY", "")

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
    **How to use**
    - Enter a ticker (e.g., AAPL).
    - Optional: enter a second ticker to compare.
    - Use normalization when comparing different-priced stocks.
    """
)

# =========================
# Data helpers
# =========================
@st.cache_data(ttl=900)
def fetch_stock_data(sym: str, period: str) -> pd.DataFrame:
    if not sym:
        return pd.DataFrame()
    df = yf.download(sym, period=period, auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()

    # Flatten MultiIndex columns if returned
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.dropna()
    return df

def add_mas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Close" in out.columns:
        out["MA20"] = out["Close"].rolling(20).mean()
        out["MA50"] = out["Close"].rolling(50).mean()
    return out

def make_line_fig(df: pd.DataFrame, title: str, y_cols: list[str]) -> go.Figure:
    fig = go.Figure()
    for c in y_cols:
        if c in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[c], mode="lines", name=c))
    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=50, b=10),
        height=420,
    )
    if use_log:
        fig.update_yaxes(type="log")
    return fig

def make_candle_fig(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"] if "Open" in df.columns else None,
        high=df["High"] if "High" in df.columns else None,
        low=df["Low"] if "Low" in df.columns else None,
        close=df["Close"] if "Close" in df.columns else None,
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
        margin=dict(l=10, r=10, t=50, b=10),
        height=520,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    if use_log:
        fig.update_yaxes(type="log")
    return fig

def compute_metrics(df: pd.DataFrame) -> dict:
    if df is None or df.empty or "Close" not in df.columns:
        return {}
    close = df["Close"].dropna()
    if len(close) < 2:
        return {}
    last = float(close.iloc[-1])
    prev = float(close.iloc[-2])
    day_pct = (last / prev - 1) * 100 if prev else 0.0

    # period change
    first = float(close.iloc[0])
    period_pct = (last / first - 1) * 100 if first else 0.0

    # simple vol estimate (std of daily returns)
    rets = close.pct_change().dropna()
    vol = float(rets.std() * 100) if len(rets) > 5 else float("nan")

    return {
        "Last Close": last,
        "Day %": day_pct,
        "Period %": period_pct,
        "Vol (daily std %)": vol,
        "Last Date": df.index[-1].date(),
    }

# =========================
# Load data
# =========================
df1 = fetch_stock_data(symbol, range_option)
df2 = fetch_stock_data(compare_symbol, range_option) if compare_symbol else pd.DataFrame()

if show_ma and not df1.empty:
    df1 = add_mas(df1)
if show_ma and not df2.empty:
    df2 = add_mas(df2)

# =========================
# Price Analysis section
# =========================
st.markdown("## Price Analysis")

if df1.empty:
    st.warning("Could not load data for that symbol. Please check the ticker and try again.")
else:
    m = compute_metrics(df1)
    if m:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Last Close", f"${m['Last Close']:.2f}")
        c2.metric("Day Change", f"{m['Day %']:.2f}%")
        c3.metric("Period Change", f"{m['Period %']:.2f}%")
        c4.metric("Daily Vol (std)", f"{m['Vol (daily std %)']:.2f}%" if m["Vol (daily std %)"] == m["Vol (daily std %)"] else "n/a")

    if compare_symbol and not df2.empty:
        # Comparison chart (raw or normalized)
        combined = pd.concat(
            [df1["Close"].rename(symbol), df2["Close"].rename(compare_symbol)],
            axis=1
        ).dropna()

        st.subheader(f"{symbol} vs {compare_symbol}")

        if normalize_compare:
            combined = (combined / combined.iloc[0] - 1) * 100
            fig = make_line_fig(combined, "Comparison (Normalized % Change)", [symbol, compare_symbol])
            fig.update_yaxes(title_text="Percent change (%)")
        else:
            fig = make_line_fig(combined, "Comparison (Raw Close Price)", [symbol, compare_symbol])
            fig.update_yaxes(title_text="Price")

        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Show each ticker’s own price chart"):
            st.plotly_chart(
                make_line_fig(df1, f"{symbol} Close", ["Close"] + (["MA20", "MA50"] if show_ma else [])),
                use_container_width=True
            )
            st.plotly_chart(
                make_line_fig(df2, f"{compare_symbol} Close", ["Close"] + (["MA20", "MA50"] if show_ma else [])),
                use_container_width=True
            )

    else:
        # Single stock view
        if chart_type == "Line":
            y_cols = ["Close"] + (["MA20", "MA50"] if show_ma else [])
            fig = make_line_fig(df1, f"{symbol} Close", y_cols)
            fig.update_yaxes(title_text="Price")
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Candlestick
            needed = {"Open", "High", "Low", "Close"}
            if not needed.issubset(set(df1.columns)):
                st.info("Candlestick view needs Open/High/Low/Close columns; falling back to line chart.")
                y_cols = ["Close"] + (["MA20", "MA50"] if show_ma else [])
                st.plotly_chart(make_line_fig(df1, f"{symbol} Close", y_cols), use_container_width=True)
            else:
                st.plotly_chart(make_candle_fig(df1, f"{symbol} Candlestick"), use_container_width=True)

        with st.expander("Show OHLC lines (optional)"):
            ohlc_cols = [c for c in ["Open", "High", "Low", "Close"] if c in df1.columns]
            if ohlc_cols:
                st.plotly_chart(make_line_fig(df1, f"{symbol} OHLC", ohlc_cols), use_container_width=True)
            else:
                st.info("OHLC columns not available for this symbol/time range.")

# =========================
# Correlation section
# =========================
st.markdown("## Cross-Stock Correlation")

corr_input = st.text_input("Symbols (comma separated)", "AAPL, MSFT, GOOGL")
corr_syms = [s.strip().upper() for s in corr_input.split(",") if s.strip()]

corr_data = {}
for s in corr_syms:
    tmp = fetch_stock_data(s, range_option)
    if not tmp.empty and "Close" in tmp.columns:
        corr_data[s] = tmp["Close"]

if corr_data:
    corr_df = pd.DataFrame(corr_data).dropna()
    if not corr_df.empty:
        fig, ax = plt.subplots()
        sns.heatmap(corr_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.info("Not enough overlapping data to compute correlations.")
else:
    st.info("Enter at least one valid symbol.")

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

# =========================
# Insider Trading Snapshot (Finviz scrape)
# =========================
st.markdown("## Insider Trading Snapshot")

@st.cache_data(ttl=3600)
def fetch_insider_finviz(sym: str) -> pd.DataFrame:
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

insider_df = fetch_insider_finviz(symbol) if symbol else pd.DataFrame()
if insider_df.empty:
    st.info("No insider data found (or the source blocked the request).")
else:
    st.dataframe(insider_df, use_container_width=True)

# =========================
# Event-Driven Insights (NewsData.io)
# =========================
st.markdown("## Event-Driven Insights")

@st.cache_data(ttl=900)
def fetch_news(sym: str):
    if not NEWS_API_KEY:
        return []
    url = f"https://newsdata.io/api/1/news?apikey={NEWS_API_KEY}&q={sym}&language=en"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        if data.get("results"):
            return data["results"][:6]
        return []
    except Exception:
        return []

news_items = fetch_news(symbol) if symbol else []
if news_items:
    for item in news_items:
        title = item.get("title", "(no title)")
        link = item.get("link", "")
        pub = item.get("pubDate", "")
        st.markdown(f"**{title}**")
        if link:
            st.markdown(f"{pub} — [Link]({link})")
        else:
            st.markdown(f"{pub}")
        st.write("")
else:
    st.info("No recent news found.")

# =========================
# Government Trades (Nancy Pelosi, etc.) — optional integration
# =========================
st.markdown("## Government Trades (Optional)")

st.write(
    "If you want weekly trades for specific politicians (e.g., Nancy Pelosi), the most reliable approach is using a dedicated Congress-trading data API.\n\n"
    "- Best path: **Quiver Quantitative API** (requires an API token). Their Python package supports `congress_trading()` and filtering by politician. :contentReference[oaicite:2]{index=2}\n\n"
    "This section will activate if you add `QUIVER_API_KEY` in Streamlit Secrets."
)

with st.expander("Enable Congress trades (advanced)"):
    st.caption("Add QUIVER_API_KEY to Streamlit Secrets, then use the controls below.")
    politicians = st.multiselect(
        "Key traders to track",
        ["Nancy Pelosi", "Marjorie Taylor Greene", "J. D. Vance"],
        default=["Nancy Pelosi"]
    )
    days_back = st.slider("Lookback window (days)", 7, 30, 7)

def _fetch_quiver_congress_trades(api_key: str) -> pd.DataFrame:
    """
    Placeholder implementation:
    - Quiver provides a Python package and API token-based access for Congress Trading. :contentReference[oaicite:3]{index=3}
    - To keep this app stable without extra dependencies, we don’t import quiverquant here by default.
    - If you want it enabled, install `quiverquant` and use the code in the comment below.
    """
    return pd.DataFrame()

if QUIVER_API_KEY:
    st.success("QUIVER_API_KEY detected. Next step: enable Quiver fetch code (see notes below).")
    st.code(
        """# Enable Quiver congress trades:
# 1) pip install quiverquant
# 2) Then add this import + function to your app:

# import quiverquant
# quiver = quiverquant.quiver(st.secrets["QUIVER_API_KEY"])
# df_congress = quiver.congress_trading()  # recent trades
# # Optional: filter by politician name:
# df_pelosi = quiver.congress_trading("Nancy Pelosi", politician=True)

# Then filter df_congress by last X days and selected politicians and display with st.dataframe().
""",
        language="python"
    )
else:
    st.info("QUIVER_API_KEY not set. This section will remain informational for now.")

# =========================
# Footer
# =========================
st.markdown("---")
st.markdown(
    "<p style='font-size:13px;color:#777'>Portfolio project: Python, Streamlit, yfinance, Plotly, seaborn, requests, BeautifulSoup.</p>",
    unsafe_allow_html=True,
)
