import os
import requests
import yfinance as yf
import pandas as pd
import plotly.express as px
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Stock Insight Dashboard",
    layout="wide"
)

NEWS_API_KEY = "pub_b94aa5942cf24b209ce26666d30b5207"

# ---------- HEADER ----------
st.markdown(
    """
    <h1 style="margin-bottom:0">Stock Insight Dashboard</h1>
    <p style="color:#555;font-size:16px;margin-top:4px">
        Interactive stock analysis tool for comparing performance, correlations,
        insider activity, and market news.
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# ---------- INPUT AREA ----------
col_left, col_right = st.columns([2, 1])

with col_left:
    symbol = st.text_input("Primary stock symbol", "AAPL").upper().strip()
    compare_symbol = st.text_input("Compare with another symbol (optional)", "").upper().strip()

with col_right:
    range_option = st.selectbox(
        "Time range",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd", "max"],
        index=2,
    )

# ---------- DATA FETCH ----------
def fetch_stock_data(sym):
    if not sym:
        return pd.DataFrame()
    df = yf.download(sym, period=range_option)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df

df = fetch_stock_data(symbol)

# =========================================================
# PRICE ANALYSIS
# =========================================================
st.markdown("## Price Analysis")

if df.empty:
    st.warning("Could not load data for that symbol.")
else:

    # SINGLE STOCK VIEW
    if not compare_symbol:

        st.subheader(f"{symbol} Closing Price")

        st.plotly_chart(
            px.line(df, x=df.index, y="Close"),
            use_container_width=True,
        )

        st.subheader("OHLC")
        st.plotly_chart(
            px.line(df, x=df.index, y=["Open", "High", "Low", "Close"]),
            use_container_width=True,
        )

    # COMPARISON VIEW
    else:

        df2 = fetch_stock_data(compare_symbol)

        if df2.empty:
            st.warning(f"Could not load comparison symbol: {compare_symbol}")
        else:

            st.subheader(f"Comparing {symbol} vs {compare_symbol}")

            combined = pd.concat(
                [df["Close"].rename(symbol), df2["Close"].rename(compare_symbol)],
                axis=1
            ).dropna()

            # NORMALIZATION TOGGLE
            show_pct = st.toggle(
                "Normalize to percentage change (compare performance instead of price)",
                value=True
            )

            if show_pct:
                combined = (combined / combined.iloc[0] - 1) * 100
                st.caption("Performance indexed to starting date (0%).")

            st.line_chart(combined)

            # ADVANCED OPTIONS (HIDDEN)
            with st.expander("Advanced chart options"):
                log_scale = st.checkbox("Use logarithmic scale")

                if log_scale:
                    st.write("Log scale view:")

                    fig = px.line(combined, x=combined.index, y=combined.columns)
                    fig.update_yaxes(type="log")

                    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# CORRELATION
# =========================================================
st.markdown("## Cross-Stock Correlation")

corr_input = st.text_input(
    "Symbols (comma separated)",
    "AAPL, MSFT, GOOGL",
)

corr_syms = [s.strip().upper() for s in corr_input.split(",") if s.strip()]

corr_data = {}
for sym in corr_syms:
    tmp = fetch_stock_data(sym)
    if not tmp.empty:
        corr_data[sym] = tmp["Close"]

if corr_data:
    corr_df = pd.DataFrame(corr_data).dropna()

    if not corr_df.empty:
        fig, ax = plt.subplots()
        sns.heatmap(corr_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

# =========================================================
# ANALYST SUMMARY
# =========================================================
st.markdown("## Analyst Summary")

def generate_summary(df, sym):
    if df.empty or len(df) < 2:
        return "Not enough data."

    change = df["Close"].iloc[-1] - df["Close"].iloc[0]
    pct = (change / df["Close"].iloc[0]) * 100

    text = f"{sym} moved {change:.2f} USD ({pct:.2f}%) over the selected period. "

    if pct > 5:
        text += "Momentum appears positive."
    elif pct < -5:
        text += "Recent trend is negative."
    else:
        text += "Price movement has been relatively stable."

    return text

if not df.empty:
    st.info(generate_summary(df, symbol))

# =========================================================
# INSIDER DATA (FINVIZ SCRAPE)
# =========================================================
st.markdown("## Insider Trading Snapshot")

@st.cache_data(ttl=3600)
def fetch_insider(sym):
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

        return pd.DataFrame(
            data,
            columns=[
                "Insider",
                "Relationship",
                "Date",
                "Transaction",
                "Cost",
                "Shares",
                "Value",
            ],
        )
    except:
        return pd.DataFrame()

insider_df = fetch_insider(symbol)

if insider_df.empty:
    st.info("No insider data found.")
else:
    st.dataframe(insider_df)

# =========================================================
# NEWS
# =========================================================
st.markdown("## Event-Driven Insights")

def fetch_news(sym):
    url = f"https://newsdata.io/api/1/news?apikey={NEWS_API_KEY}&q={sym}&language=en"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        if data.get("results"):
            return data["results"][:5]
    except:
        pass
    return []

news_items = fetch_news(symbol)

if news_items:
    for item in news_items:
        st.markdown(f"**{item['title']}**")
        st.markdown(f"{item['pubDate']} — [Link]({item['link']})")
        st.write("")
else:
    st.info("No recent news found.")

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.markdown(
    "<p style='font-size:13px;color:#777'>Built as a portfolio project using Python, Streamlit, yfinance, Plotly, and external APIs.</p>",
    unsafe_allow_html=True,
)
