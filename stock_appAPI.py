import os
import time
import datetime as dt
from typing import Tuple

import requests
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px


# ============================================================
# Page setup
# ============================================================
st.set_page_config(page_title="Stock Insights by Ming Quan", layout="wide")

CSS = """
<style>
  :root{
    --bg:#0b0f14;
    --panel: rgba(255,255,255,0.04);
    --panel2: rgba(255,255,255,0.06);
    --border: rgba(255,255,255,0.10);
    --text: rgba(255,255,255,0.92);
    --muted: rgba(255,255,255,0.65);
  }

  html, body, [data-testid="stAppViewContainer"], .stApp {
    background: radial-gradient(1200px 800px at 20% 0%, rgba(110,231,255,0.10), transparent 55%),
                radial-gradient(1000px 700px at 80% 10%, rgba(255,80,120,0.08), transparent 55%),
                var(--bg) !important;
    color: var(--text) !important;
  }

  [data-testid="stDecoration"] { background: transparent !important; }
  [data-testid="stHeader"] { background: transparent !important; }

  .block-container { padding-top: 1.4rem; padding-bottom: 2.7rem; max-width: 1200px; }

  .hero {
    border: 1px solid var(--border);
    background: linear-gradient(180deg, rgba(255,255,255,0.07), rgba(255,255,255,0.03));
    border-radius: 18px;
    padding: 18px 18px;
    margin-bottom: 18px;
  }

  .hero-title { font-size: 34px; font-weight: 800; margin: 0; }
  .hero-sub { color: var(--muted); margin-top: 10px; font-size: 14.5px; line-height: 1.4; }

  .section {
    border: 1px solid var(--border);
    background: var(--panel);
    border-radius: 18px;
    padding: 16px 16px;
    margin-top: 18px;
  }

  .section-title{
    font-size: 18px;
    font-weight: 750;
    margin: 0 0 12px 0;
  }

  .muted { color: var(--muted); }
  .spacer-sm { height: 10px; }
  .spacer-md { height: 16px; }
  .spacer-lg { height: 22px; }

  hr { border: none; border-top: 1px solid var(--border); margin: 18px 0; }

  .stTextInput input, .stSelectbox select {
    background: var(--panel2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 12px !important;
  }

  .stButton button {
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
    background: rgba(255,255,255,0.07) !important;
    color: var(--text) !important;
    padding: 0.55rem 1rem !important;
  }

  .stButton button:hover { border-color: rgba(110,231,255,0.35) !important; }

  div[data-testid="stDataFrame"]{
    border: 1px solid var(--border);
    border-radius: 14px;
    overflow: hidden;
  }

  a {
    color: rgba(110,231,255,0.95) !important;
    text-decoration: none;
  }

  a:hover {
    text-decoration: underline;
  }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
      <p class="hero-title">Stock Insights by Ming Quan</p>
      <div class="hero-sub">
        Price comparisons, correlations, news context, and government disclosures.
      </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# Secrets / API keys
# ============================================================
NEWS_API_KEY = (
    st.secrets.get("NEWS_API_KEY", "")
    or os.getenv("NEWS_API_KEY", "")
    or "pub_b94aa5942cf24b209ce26666d30b5207"
).strip()


# ============================================================
# Helpers
# ============================================================
def _flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.dropna()
    return df


def _normalize_pct_change(s: pd.Series) -> pd.Series:
    s = s.dropna()
    if s.empty:
        return s

    base = s.iloc[0]
    if pd.isna(base) or base == 0:
        return s * 0

    return (s / base - 1) * 100


@st.cache_data(ttl=86400, show_spinner=False, max_entries=256)
def get_company_search_terms(symbol: str) -> list[str]:
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return []

    terms = [symbol]

    try:
        info = yf.Ticker(symbol).info or {}

        short_name = str(info.get("shortName") or "").strip()
        long_name = str(info.get("longName") or "").strip()

        for name in [short_name, long_name]:
            if name and name.upper() != symbol and name not in terms:
                terms.append(name)

        suffixes = [
            ", Inc.", " Inc.", ", Inc", " Inc",
            ", Corporation", " Corporation",
            ", Corp.", " Corp.", ", Corp", " Corp",
            ", Ltd.", " Ltd.", ", Ltd", " Ltd",
            ", PLC", " PLC",
            ", Class A", " Class A",
            ", Class B", " Class B",
        ]

        for name in [short_name, long_name]:
            cleaned = name
            for suffix in suffixes:
                cleaned = cleaned.replace(suffix, "")
            cleaned = cleaned.strip()

            if cleaned and cleaned.upper() != symbol and cleaned not in terms:
                terms.append(cleaned)

    except Exception:
        pass

    deduped = []
    seen = set()

    for term in terms:
        key = term.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(term)

    return deduped[:4]


def build_news_query(symbol: str) -> str:
    terms = get_company_search_terms(symbol)

    if not terms:
        return symbol

    exact_parts = [f'"{t}"' if " " in t else t for t in terms]

    if len(exact_parts) >= 2:
        return f"({' OR '.join(exact_parts)}) AND (stock OR shares OR earnings OR market)"

    return f"{exact_parts[0]} AND (stock OR shares OR earnings OR market)"


# ============================================================
# Price Data
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False, max_entries=256)
def fetch_from_yahoo(symbol: str, period: str) -> pd.DataFrame:
    df = yf.download(
        symbol,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    return _flatten_yf_columns(df)


@st.cache_data(ttl=3600, show_spinner=False, max_entries=256)
def fetch_from_stooq(symbol: str, period: str) -> pd.DataFrame:
    sym = (symbol or "").strip().lower()
    if not sym:
        return pd.DataFrame()

    candidates = [sym]

    if not sym.endswith(".us"):
        candidates.append(f"{sym}.us")

    for candidate in candidates:
        url = f"https://stooq.com/q/d/l/?s={candidate}&i=d"

        try:
            df = pd.read_csv(url)
        except Exception:
            continue

        if df is None or df.empty or "Date" not in df.columns:
            continue

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
        df = df.rename(columns={col: col.title() for col in df.columns})

        keep = [k for k in ["Open", "High", "Low", "Close", "Volume"] if k in df.columns]
        df = df[keep].dropna()

        if period != "max":
            days = {
                "1mo": 35,
                "3mo": 100,
                "6mo": 210,
                "1y": 370,
                "2y": 740,
                "5y": 1850,
                "ytd": 370,
            }.get(period, 210)

            cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=days)
            df = df[df.index >= cutoff]

        if not df.empty:
            return df

    return pd.DataFrame()


def fetch_price_data(symbol: str, period: str) -> Tuple[pd.DataFrame, str]:
    symbol = (symbol or "").strip().upper()

    if not symbol:
        return pd.DataFrame(), "none"

    for attempt in range(3):
        try:
            df = fetch_from_yahoo(symbol, period)
            if not df.empty:
                return df, "Yahoo Finance"
        except Exception:
            time.sleep(0.6 * (attempt + 1))

    df2 = fetch_from_stooq(symbol, period)

    if not df2.empty:
        return df2, "Stooq"

    return pd.DataFrame(), "none"


# ============================================================
# NewsData.io
# ============================================================
@st.cache_data(ttl=1800, show_spinner=False, max_entries=256)
def fetch_news(symbol: str, limit: int = 8, refresh_token: int = 0) -> pd.DataFrame:
    if not NEWS_API_KEY:
        return pd.DataFrame()

    symbol = (symbol or "").strip().upper()

    if not symbol:
        return pd.DataFrame()

    query = build_news_query(symbol)

    url = "https://newsdata.io/api/1/news"
    params = {
        "apikey": NEWS_API_KEY,
        "q": query,
        "language": "en",
        "category": "business",
        "size": min(10, max(3, int(limit))),
    }

    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return pd.DataFrame()

    results = data.get("results") or []

    company_terms = [t.lower() for t in get_company_search_terms(symbol)]
    company_terms.append(symbol.lower())

    rows = []

    for item in results:
        title = item.get("title") or ""
        desc = item.get("description") or ""
        content = item.get("content") or ""
        source = item.get("source_id") or item.get("source_name") or "Unknown source"
        published = item.get("pubDate")
        link = item.get("link")
        image_url = item.get("image_url") or item.get("image")

        combined_text = f"{title} {desc} {content}".lower()

        if not any(term in combined_text for term in company_terms):
            continue

        rows.append({
            "Title": title,
            "Source": source,
            "PublishedRaw": pd.to_datetime(published, errors="coerce", utc=True),
            "Published": published,
            "Link": link,
            "Summary": desc or content,
            "Image": image_url,
        })

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df = df.drop_duplicates(subset=["Title", "Link"]).copy()
    df = df.sort_values("PublishedRaw", ascending=False)
    df["Published"] = df["PublishedRaw"].dt.strftime("%Y-%m-%d %H:%M UTC")

    return df.head(limit).reset_index(drop=True)


# ============================================================
# Capitol Trades
# ============================================================
CAPITOL_BFF_TRADES = "https://bff.capitoltrades.com/trades"


def capitol_request_with_retry(url: str, params: dict, retries: int = 6) -> dict:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; StockInsights/1.0)",
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.capitoltrades.com/",
        "Origin": "https://www.capitoltrades.com",
        "Connection": "keep-alive",
    }

    last_exc = None

    for i in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=25)

            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"{r.status_code} {r.reason}", response=r)

            r.raise_for_status()
            return r.json()

        except Exception as e:
            last_exc = e
            time.sleep(min(12.0, 0.9 * (2 ** i)))

    raise last_exc if last_exc else RuntimeError("Capitol Trades request failed.")


@st.cache_data(ttl=3600, show_spinner=False, max_entries=32)
def fetch_capitol_trades(pages: int = 1, page_size: int = 50) -> pd.DataFrame:
    rows = []

    for page in range(1, pages + 1):
        params = {"page": page, "pageSize": page_size, "per_page": page_size}
        payload = capitol_request_with_retry(CAPITOL_BFF_TRADES, params=params, retries=6)

        data = payload.get("data", []) if isinstance(payload, dict) else []

        for row in data:
            pol = row.get("politician", {}) or {}
            asset = row.get("asset", {}) or {}

            politician = f"{pol.get('firstName','')} {pol.get('lastName','')}".strip()
            ticker = asset.get("assetTicker")

            if isinstance(ticker, str):
                ticker = ticker.replace(":US", "")

            rows.append({
                "Published": pd.to_datetime(row.get("published"), errors="coerce"),
                "Trade Date": pd.to_datetime(row.get("traded"), errors="coerce"),
                "Politician": politician,
                "Party": pol.get("party"),
                "Chamber": pol.get("chamber"),
                "State": pol.get("_stateId"),
                "Ticker": ticker,
                "Asset": asset.get("assetName"),
                "Action": row.get("type"),
                "Size": row.get("size"),
                "Delay (days)": row.get("filedAfter"),
                "Source": row.get("sourceUrl") or row.get("url") or None,
            })

        time.sleep(0.2)

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    return df.sort_values(by=["Published", "Trade Date"], ascending=False)


# ============================================================
# Session state
# ============================================================
if "df1" not in st.session_state:
    st.session_state.df1 = pd.DataFrame()
    st.session_state.df2 = pd.DataFrame()
    st.session_state.last_fetch = None

if "last_good_trades" not in st.session_state:
    st.session_state.last_good_trades = pd.DataFrame()
    st.session_state.last_good_trades_ts = None

if "news_refresh_counter" not in st.session_state:
    st.session_state.news_refresh_counter = 0


# ============================================================
# Controls
# ============================================================
colA, colB = st.columns([1.1, 1.1], gap="large")

with colA:
    symbol = st.text_input("Primary stock", "AAPL").strip().upper()

with colB:
    compare_symbol = st.text_input("Optional second stock (comparison)", "").strip().upper()

range_option = st.selectbox(
    "Time range",
    ["1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd", "max"],
    index=2,
)

opt1, opt2, opt3 = st.columns([1.0, 1.0, 1.2], gap="large")

with opt1:
    fetch_btn = st.button("Fetch / Refresh Data", type="primary")

with opt2:
    compare_as_pct = st.checkbox(
        "Compare as percent change",
        value=True,
        disabled=not bool(compare_symbol),
    )

with opt3:
    with st.expander("Advanced"):
        use_log = st.checkbox(
            "Use log scale",
            value=False,
            disabled=not bool(compare_symbol),
        )

if fetch_btn:
    with st.spinner("Fetching price data..."):
        df1, _ = fetch_price_data(symbol, range_option)

        df2 = pd.DataFrame()
        if compare_symbol:
            df2, _ = fetch_price_data(compare_symbol, range_option)

        st.session_state.df1 = df1
        st.session_state.df2 = df2
        st.session_state.last_fetch = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

df1 = st.session_state.df1
df2 = st.session_state.df2


# ============================================================
# Price Charts
# ============================================================
st.markdown('<div class="section"><div class="section-title">Price charts</div>', unsafe_allow_html=True)
st.markdown('<div class="spacer-sm"></div>', unsafe_allow_html=True)

if st.session_state.last_fetch:
    st.markdown(
        f'<div class="muted">Last refresh: {st.session_state.last_fetch}</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="spacer-md"></div>', unsafe_allow_html=True)

st.markdown(
    '<div class="muted">Data: Yahoo Finance with Stooq fallback.</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="spacer-lg"></div>', unsafe_allow_html=True)


def plot_compare_chart(
    series_a: pd.Series,
    name_a: str,
    series_b: pd.Series,
    name_b: str,
    pct_mode: bool,
) -> go.Figure:
    combined = pd.concat(
        [series_a.rename(name_a), series_b.rename(name_b)],
        axis=1,
    ).dropna()

    if pct_mode:
        combined = combined.apply(_normalize_pct_change, axis=0)
        y_title = "Percent change (%)"
    else:
        y_title = "Price"

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=combined.index,
            y=combined[name_a],
            mode="lines",
            name=name_a,
            line=dict(color="red", width=2.6),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=combined.index,
            y=combined[name_b],
            mode="lines",
            name=name_b,
            line=dict(color="royalblue", width=2.6),
        )
    )

    fig.update_layout(
        template="plotly_dark",
        title=dict(text=f"{name_a} vs {name_b}", x=0.0, xanchor="left", font=dict(size=18)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=100, b=10),
        height=480,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            x=1.0,
            xanchor="right",
            y=1.20,
            yanchor="top",
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
    )

    fig.update_yaxes(title_text=y_title, showgrid=True, gridcolor="rgba(255,255,255,0.08)")
    fig.update_xaxes(showgrid=False)

    if (not pct_mode) and use_log:
        fig.update_yaxes(type="log")

    return fig


def plot_single_close(df: pd.DataFrame, sym: str) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            name=sym,
            line=dict(color="royalblue", width=2.6),
        )
    )

    fig.update_layout(
        template="plotly_dark",
        title=dict(text=f"{sym} closing price", x=0.0, xanchor="left", font=dict(size=18)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=95, b=10),
        height=480,
        hovermode="x unified",
        showlegend=False,
    )

    fig.update_yaxes(title_text="Price", showgrid=True, gridcolor="rgba(255,255,255,0.08)")
    fig.update_xaxes(showgrid=False)

    return fig


def plot_candles(df: pd.DataFrame, sym: str) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name=sym,
        )
    )

    fig.update_layout(
        template="plotly_dark",
        title=dict(text=f"{sym} OHLC candlestick", x=0.0, xanchor="left", font=dict(size=18)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=95, b=10),
        height=580,
        hovermode="x unified",
        showlegend=False,
    )

    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
    fig.update_xaxes(showgrid=False)

    return fig


if df1.empty:
    st.warning("No data loaded yet. Click Fetch / Refresh Data.")
else:
    if compare_symbol and not df2.empty and "Close" in df2.columns:
        st.plotly_chart(
            plot_compare_chart(df1["Close"], symbol, df2["Close"], compare_symbol, compare_as_pct),
            width="stretch",
        )
    else:
        st.plotly_chart(plot_single_close(df1, symbol), width="stretch")
        st.markdown('<div class="spacer-md"></div>', unsafe_allow_html=True)

        if set(["Open", "High", "Low", "Close"]).issubset(df1.columns):
            st.plotly_chart(plot_candles(df1, symbol), width="stretch")

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# Event-driven Insights
# ============================================================
st.markdown('<div class="section"><div class="section-title">Event-driven insights</div>', unsafe_allow_html=True)
st.markdown('<div class="spacer-sm"></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="muted">Recent business headlines related to the selected stock.</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="spacer-lg"></div>', unsafe_allow_html=True)

news_cols = st.columns([1.2, 0.8, 0.8], gap="large")

with news_cols[0]:
    news_symbol = st.text_input("News search symbol", value=symbol).strip().upper()

with news_cols[1]:
    news_limit = st.selectbox("Articles", [5, 8, 10], index=1)

with news_cols[2]:
    refresh_news = st.button("Refresh news")

if refresh_news:
    st.session_state.news_refresh_counter += 1

if not NEWS_API_KEY:
    st.info("News is available when a NewsData.io key is added in Streamlit secrets as NEWS_API_KEY.")
else:
    try:
        with st.spinner("Fetching news..."):
            news_df = fetch_news(
                news_symbol,
                limit=int(news_limit),
                refresh_token=st.session_state.news_refresh_counter,
            )

        if news_df.empty:
            st.warning("No relevant news returned for that stock right now.")
        else:
            for _, row in news_df.iterrows():
                title = row.get("Title") or ""
                link = row.get("Link") or ""
                source = row.get("Source") or "Unknown source"
                published = row.get("Published") or ""
                summary = row.get("Summary") or ""
                image = row.get("Image") or ""

                article_cols = st.columns([0.24, 0.76], gap="medium")

                with article_cols[0]:
                    if image:
                        try:
                            st.image(image, use_container_width=True)
                        except Exception:
                            pass

                with article_cols[1]:
                    if link:
                        st.markdown(f"**[{title}]({link})**")
                    else:
                        st.markdown(f"**{title}**")

                    st.markdown(
                        f"<span class='muted'>{source} • {published}</span>",
                        unsafe_allow_html=True,
                    )

                    if summary:
                        trimmed = summary[:280] + ("..." if len(summary) > 280 else "")
                        st.markdown(
                            f"<span class='muted'>{trimmed}</span>",
                            unsafe_allow_html=True,
                        )

                st.markdown("<div class='spacer-md'></div>", unsafe_allow_html=True)

    except Exception as e:
        st.warning(f"News is temporarily unavailable: {e}")

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# Government trading activity
# ============================================================
st.markdown('<div class="section"><div class="section-title">Government trading activity</div>', unsafe_allow_html=True)
st.markdown('<div class="spacer-sm"></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="muted">Public trade disclosures aggregated from Capitol Trades.</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="spacer-lg"></div>', unsafe_allow_html=True)

filters1, filters2, filters3 = st.columns([1.1, 0.9, 1.0], gap="large")

with filters1:
    trader_query = st.text_input("Filter by politician name", "").strip()

with filters2:
    action_filter = st.selectbox("Action", ["All", "Buy", "Sell"], index=0)

with filters3:
    lookback_days = st.selectbox("Lookback window", [7, 14, 30, 90], index=2)

more1, more2, more3 = st.columns([1, 1, 1], gap="large")

with more1:
    pages = st.slider("Pages", 1, 3, 1)

with more2:
    rows_to_show = st.slider("Rows to show", 10, 100, 25)

with more3:
    refresh_trades = st.button("Refresh government trades")

should_fetch_trades = refresh_trades or st.session_state.last_good_trades.empty

try:
    trades = pd.DataFrame()

    if should_fetch_trades:
        with st.spinner("Fetching government trades..."):
            trades = fetch_capitol_trades(pages=int(pages), page_size=50)

        if not trades.empty:
            st.session_state.last_good_trades = trades
            st.session_state.last_good_trades_ts = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        else:
            trades = st.session_state.last_good_trades
    else:
        trades = st.session_state.last_good_trades

    if trades is None or trades.empty:
        st.warning("Capitol Trades is currently unavailable. Try again later.")
    else:
        if st.session_state.last_good_trades_ts:
            st.markdown(
                f"<div class='muted'>Last successful refresh: {st.session_state.last_good_trades_ts}</div>",
                unsafe_allow_html=True,
            )
            st.markdown("<div class='spacer-md'></div>", unsafe_allow_html=True)

        cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=int(lookback_days))

        trades_f = trades.copy()
        trades_f = trades_f[trades_f["Published"].fillna(pd.Timestamp.min) >= cutoff]

        if trader_query:
            trades_f = trades_f[
                trades_f["Politician"].fillna("").str.contains(trader_query, case=False, na=False)
            ]

        if action_filter != "All":
            trades_f = trades_f[
                trades_f["Action"].fillna("").str.contains(action_filter, case=False, na=False)
            ]

        display_cols = [
            "Published",
            "Trade Date",
            "Politician",
            "Party",
            "Chamber",
            "State",
            "Ticker",
            "Asset",
            "Action",
            "Size",
            "Delay (days)",
            "Source",
        ]

        trades_f = trades_f[display_cols].copy()

        for c in ["Published", "Trade Date"]:
            trades_f[c] = pd.to_datetime(trades_f[c], errors="coerce").dt.strftime("%Y-%m-%d")

        st.markdown(
            """
            <div class="muted">
              “Published” is when the disclosure appeared. “Trade Date” is when the trade occurred.
              “Delay (days)” is filing lag. “Size” is the reported range.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div class='spacer-md'></div>", unsafe_allow_html=True)
        st.dataframe(trades_f.head(int(rows_to_show)), width="stretch")

        tickers = trades_f["Ticker"].dropna()
        tickers = tickers[tickers.astype(str).str.len() > 0]

        if not tickers.empty:
            st.markdown("<div class='spacer-md'></div>", unsafe_allow_html=True)

            top = tickers.value_counts().head(10).reset_index()
            top.columns = ["Ticker", "Mentions"]

            bar = px.bar(top, x="Ticker", y="Mentions")

            bar.update_layout(
                template="plotly_dark",
                title=dict(text="Most mentioned tickers", x=0.0, xanchor="left", font=dict(size=16)),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=90, b=10),
                height=360,
            )

            st.plotly_chart(bar, width="stretch")

except Exception as e:
    cached = st.session_state.last_good_trades

    if cached is not None and not cached.empty:
        st.warning(f"Capitol Trades temporarily unavailable. Showing cached results. Error: {e}")
        st.dataframe(cached.head(int(rows_to_show)), width="stretch")
    else:
        st.warning(f"Government trade data temporarily unavailable: {e}")

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# Cross-stock correlation explorer
# ============================================================
st.markdown('<div class="section"><div class="section-title">Cross-stock correlation explorer</div>', unsafe_allow_html=True)
st.markdown('<div class="spacer-sm"></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="muted">Correlation uses daily returns over the selected window.</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="spacer-lg"></div>', unsafe_allow_html=True)

corr_cols = st.columns([1.3, 0.9, 0.8], gap="large")

with corr_cols[0]:
    corr_input = st.text_input("Symbols comma-separated", "AAPL,MSFT,GOOGL").strip()

with corr_cols[1]:
    corr_period = st.selectbox("Correlation range", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)

with corr_cols[2]:
    run_corr = st.button("Run correlation")

if run_corr:
    syms = [s.strip().upper() for s in corr_input.split(",") if s.strip()][:10]
    closes = {}

    with st.spinner("Fetching symbols..."):
        for s in syms:
            dfx, _src = fetch_price_data(s, corr_period)

            if not dfx.empty and "Close" in dfx.columns:
                closes[s] = dfx["Close"]

    if len(closes) < 2:
        st.warning("Not enough symbols loaded to compute correlation.")
    else:
        corr_df = pd.DataFrame(closes).dropna()
        corr = corr_df.pct_change().dropna().corr()

        heat = px.imshow(corr, text_auto=True, aspect="auto")

        heat.update_layout(
            template="plotly_dark",
            title=dict(text="Correlation matrix", x=0.0, xanchor="left", font=dict(size=16)),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=90, b=10),
            height=560,
        )

        st.plotly_chart(heat, width="stretch")

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# Contact
# ============================================================
st.markdown("<hr/>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="section">
      <div class="section-title">Contact</div>
      <div class="muted">
        Email: <a href="mailto:mingquan916@gmail.com">mingquan916@gmail.com</a><br/>
        LinkedIn: <a href="https://www.linkedin.com/in/mingquandata/" target="_blank">linkedin.com/in/mingquandata</a>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
