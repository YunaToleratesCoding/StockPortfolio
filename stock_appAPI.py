import os
import requests
import yfinance as yf
import pandas as pd
import plotly.express as px
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO
import datetime

# API Keys
NEWS_API_KEY = "pub_b94aa5942cf24b209ce26666d30b5207"  # Replace with your actual News API key
FMP_API_KEY = "fjU7e0CFC4drXEAbHNQMOAzseiEBty9B"  # Replace with your FinancialModelingPrep key

st.set_page_config(page_title="Stock Insight Dashboard", layout="wide")
st.title("\U0001F4C8 Stock Insight Dashboard")

symbol = st.text_input("Enter a stock symbol (e.g., AAPL):", "AAPL").upper()
compare_symbol = st.text_input("Compare with another symbol (optional):").upper()

# Time range filter
range_option = st.selectbox("Select time range:", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd", "max"], index=2)

# Fetch stock data
def fetch_stock_data(symbol):
    return yf.download(symbol, period=range_option)

# Display price chart
def show_price_chart(df, symbol):
    fig = px.line(df, x=df.index, y="Close", title=f"Closing Prices for {symbol}")
    st.plotly_chart(fig, use_container_width=True)

# Plotting
if not compare_symbol:
    df = fetch_stock_data(symbol)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    if not df.empty:
        st.subheader(f"\U0001F4C8 Price Chart for {symbol}")
        st.plotly_chart(px.line(df, x=df.index, y="Close", title=f"{symbol} Closing Prices"), use_container_width=True)

        st.subheader("\U0001F50D OHLC Chart")
        fig = px.line(df, x=df.index, y=["Open", "High", "Low", "Close"], title=f"OHLC for {symbol}")
        st.plotly_chart(fig, use_container_width=True)

else:
    df1 = fetch_stock_data(symbol)
    df2 = fetch_stock_data(compare_symbol)

    if isinstance(df1.columns, pd.MultiIndex):
        df1.columns = [col[0] for col in df1.columns]
    if isinstance(df2.columns, pd.MultiIndex):
        df2.columns = [col[0] for col in df2.columns]

    if not df1.empty and not df2.empty:
        st.subheader(f"\U0001F4C8 Comparing {symbol} vs {compare_symbol}")
        combined = pd.concat([df1["Close"], df2["Close"]], axis=1)
        combined.columns = [symbol, compare_symbol]
        st.line_chart(combined)

# Cross-stock correlation
st.subheader("\U0001F517 Cross-Stock Correlation")
correlation_symbols = st.text_input("Enter multiple symbols to check correlation (comma separated):", "AAPL,MSFT,GOOGL")
correlation_list = [s.strip().upper() for s in correlation_symbols.split(",") if s.strip()]

correlation_data = {}
for sym in correlation_list:
    try:
        data = fetch_stock_data(sym)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
        correlation_data[sym] = data["Close"]
    except Exception as e:
        st.warning(f"Could not fetch data for {sym}: {e}")

if correlation_data:
    corr_df = pd.DataFrame(correlation_data).dropna()
    st.write("Correlation Matrix (Closing Prices):")
    fig, ax = plt.subplots()
    sns.heatmap(corr_df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Narrative summary
def generate_narrative(df, symbol):
    recent_change = df["Close"].iloc[-1] - df["Close"].iloc[0]
    pct_change = (recent_change / df["Close"].iloc[0]) * 100
    summary = f"{symbol} changed {recent_change:.2f} USD ({pct_change:.2f}%) over the selected period."
    if pct_change > 5:
        summary += " This may indicate strong investor confidence."
    elif pct_change < -5:
        summary += " This decline might be due to sector rotation or recent negative news."
    else:
        summary += " The stock remained relatively stable."
    return summary

if 'df' in locals() and not df.empty:
    st.subheader("\U0001F9E0 Analyst-Style Summary")
    st.info(generate_narrative(df, symbol))

# Insider and institutional activity (FMP API)
st.subheader("\U0001F3DB\ufe0f Insider & Institutional Activity")

def fetch_insider_and_institutional(symbol):
    insider_url = f"https://financialmodelingprep.com/api/v4/insider-trading?symbol={symbol}&apikey={FMP_API_KEY}"
    institution_url = f"https://financialmodelingprep.com/api/v3/institutional-holder/{symbol}?apikey={FMP_API_KEY}"
    try:
        insider_resp = requests.get(insider_url).json()
        institution_resp = requests.get(institution_url).json()

        if isinstance(insider_resp, list) and insider_resp:
            st.markdown("**Recent Insider Trades:**")
            for trade in insider_resp[:5]:
                st.markdown(f"- {trade['transactionDate']} - {trade['reportingCik']} - {trade['transactionType']} {trade['securitiesTransacted']} shares")

        if isinstance(institution_resp, list) and institution_resp:
            st.markdown("**Major Institutional Holders:**")
            for holder in institution_resp[:5]:
                st.markdown(f"- {holder['holder']} holds {holder['shares']} shares")

    except Exception as e:
        st.warning(f"Error fetching insider/institutional data: {e}")

if FMP_API_KEY != "YOUR_FMP_API_KEY":
    fetch_insider_and_institutional(symbol)
else:
    st.info("Add your FMP API key to show insider and institutional data.")

# Event-driven news
st.subheader("\U0001F4F0 Event-Driven Insights")

def fetch_news(symbol):
    if NEWS_API_KEY == "YOUR_API_KEY_HERE":
        return "No API key provided. News will not be displayed."

    url = f"https://newsdata.io/api/1/news?apikey={NEWS_API_KEY}&q={symbol}&language=en"
    try:
        response = requests.get(url)
        data = response.json()
        if data.get("status") == "success" and data.get("results"):
            return data["results"][:5]
        else:
            return []
    except Exception as e:
        return f"Error fetching news: {e}"

news_data = fetch_news(symbol)
if isinstance(news_data, list):
    for item in news_data:
        st.markdown(f"**{item['title']}**")
        st.markdown(f"{item['pubDate']} - [{item['link']}]({item['link']})")
        st.write("")
elif isinstance(news_data, str):
    st.warning(news_data)
else:
    st.info("No recent news found.")
