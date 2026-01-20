import warnings

warnings.filterwarnings('ignore')

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
import feedparser
from textblob import TextBlob
import time
import urllib.parse
import os
import re

# --- CONFIGURATION ---
st.set_page_config(page_title="CSE Command Center", layout="wide", page_icon="🛡️")

if 'portfolio' not in st.session_state: st.session_state['portfolio'] = []
if 'news_cache' not in st.session_state: st.session_state['news_cache'] = {}
if 'fin_cache' not in st.session_state: st.session_state['fin_cache'] = {}

PORTFOLIO_FILE = 'my_portfolio.csv'


# --- BACKEND API ---
class CSEBackend:
    def __init__(self):
        self.base_url = "https://www.cse.lk/api/"
        self.headers = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"}

    def _post(self, endpoint, payload=None):
        try:
            r = requests.post(f"{self.base_url}{endpoint}", data=payload, headers=self.headers, timeout=10)
            r.raise_for_status()
            return r.json()
        except:
            return {}

    @st.cache_data(ttl=60)
    def get_market_summary(_self):
        aspi = _self._post("aspiData")
        aspi_val = aspi.get('reqAspiData', {}).get('value', 0) if isinstance(aspi, dict) else 0
        aspi_pct = aspi.get('reqAspiData', {}).get('changePercentage', 0) if isinstance(aspi, dict) else 0

        snp = _self._post("snpData")
        snp_val = snp.get('reqSnpData', {}).get('value', 0) if isinstance(snp, dict) else 0
        snp_pct = snp.get('reqSnpData', {}).get('changePercentage', 0) if isinstance(snp, dict) else 0

        stats = _self._post("marketSummery")
        turnover = (stats.get('tradeVolume') or 0) / 1_000_000 if isinstance(stats, dict) else 0
        return aspi_val, aspi_pct, snp_val, snp_pct, turnover

    @st.cache_data(ttl=300)
    def get_all_stocks(_self):
        data = _self._post("tradeSummary")
        if isinstance(data, dict) and 'reqTradeSummery' in data:
            return pd.DataFrame(data['reqTradeSummery'])
        return pd.DataFrame()

    def get_chart_data(self, symbol, period_id):
        info_resp = self._post("companyInfoSummery", {"symbol": symbol})
        stock_id = info_resp.get('reqSymbolInfo', {}).get('id') if info_resp else None
        name = info_resp.get('reqSymbolInfo', {}).get('name') if info_resp else symbol

        if not stock_id: return None, []

        payload = {"stockId": stock_id, "period": str(period_id)}
        chart_resp = self._post("companyChartDataByStock", payload)
        return name, chart_resp.get('chartData', [])


# --- PORTFOLIO MANAGER ---
def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        return pd.read_csv(PORTFOLIO_FILE)
    return pd.DataFrame(columns=['Stock', 'Buy Price', 'Quantity'])


def save_portfolio(df):
    df.to_csv(PORTFOLIO_FILE, index=False)


def add_to_portfolio(stock, price, qty):
    df = load_portfolio()
    df = df[df['Stock'] != stock]
    new_row = pd.DataFrame([{'Stock': stock, 'Buy Price': float(price), 'Quantity': int(qty)}])
    df = pd.concat([df, new_row], ignore_index=True)
    save_portfolio(df)


def clear_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        os.remove(PORTFOLIO_FILE)


# --- ENGINE 7: PORTFOLIO GUARDIAN ---
def check_portfolio_health(portfolio_df, display_df):
    api = CSEBackend()
    report = []
    progress_bar = st.progress(0)
    total_stocks = len(portfolio_df)

    for i, row in portfolio_df.iterrows():
        stock = row['Stock']
        buy_price = row['Buy Price']
        qty = row['Quantity']

        try:
            curr_data = display_df[display_df['Code'] == stock].iloc[0]
            curr_price = float(curr_data['Price'])
            company_name = curr_data['Company']
        except:
            curr_price = 0
            company_name = stock

        if curr_price <= 0:
            report.append({
                "Stock": stock, "Profit/Loss %": 0.0, "Current Price": 0.0,
                "Recommendation": "⚠️ DATA", "Reason": "Price Unavailable", "Trend": []
            })
            continue

        _, raw = api.get_chart_data(stock, 5)
        trend_broken = False
        chart_trend = []

        if raw:
            chart_df = pd.DataFrame(raw)
            chart_df['p'] = pd.to_numeric(chart_df['p'], errors='coerce')
            avg_30 = chart_df['p'].tail(30).mean()
            if curr_price < avg_30: trend_broken = True
            chart_trend = chart_df['p'].tail(15).tolist()

        sent_score, sent_summary, _ = check_news_sentiment(company_name)

        pl_pct = ((curr_price - buy_price) / buy_price) * 100
        rec = "🛡️ HOLD";
        reason = "Normal";
        color = "white"

        if sent_score < 0:
            rec = "🚨 EXIT";
            reason = f"Bad News"
        elif pl_pct <= -10.0:
            rec = "✂️ CUT";
            reason = "Stop Loss"
        elif pl_pct >= 20.0:
            rec = "💰 PROFIT";
            reason = "Target Hit"
        elif trend_broken and pl_pct > 0:
            rec = "📉 EXIT";
            reason = "Trend Break"

        report.append({
            "Stock": stock, "Buy Price": buy_price, "Current Price": curr_price,
            "Profit/Loss %": pl_pct / 100,
            "Recommendation": rec, "Reason": reason, "Trend": chart_trend
        })
        progress_bar.progress((i + 1) / total_stocks)

    progress_bar.empty()
    return pd.DataFrame(report)


# --- HELPER ENGINES ---
def run_heavy_monte_carlo(df, days=30, simulations=50000):
    if len(df) < 50: return None
    prices = df['p'].values;
    prices = prices[prices > 0]
    if len(prices) < 10: return None

    log_returns = np.log(prices[1:] / prices[:-1])
    u = np.mean(log_returns);
    var = np.var(log_returns)
    drift = u - (0.5 * var);
    stdev = np.std(log_returns)

    Z = np.random.normal(0, 1, (simulations, days))
    daily_returns = np.exp(drift + stdev * Z)

    price_paths = np.vstack([np.ones(simulations) * prices[-1], daily_returns.T]).T
    price_paths = np.cumprod(price_paths, axis=1)
    final_prices = price_paths[:, -1]

    expected_price = np.mean(final_prices)
    upside_pct = ((expected_price - prices[-1]) / prices[-1]) * 100
    win_prob = (np.sum(final_prices > prices[-1]) / simulations) * 100

    return {"Upside": upside_pct, "WinRate": win_prob, "ExpectedPrice": expected_price}


def run_jump_stress_test(df, days=30, simulations=10000):
    if len(df) < 50: return None
    try:
        prices = df['p'].values;
        prices = prices[prices > 0]
        log_returns = np.log(prices[1:] / prices[:-1])
        mu = np.mean(log_returns);
        sigma = np.std(log_returns)

        jump_threshold = max(3 * sigma, 0.02)
        jumps = log_returns[np.abs(log_returns) > jump_threshold]
        lambda_j = len(jumps) / len(log_returns) if len(jumps) > 0 else 0
        mu_j = np.mean(jumps) if len(jumps) > 0 else 0
        sigma_j = np.std(jumps) if len(jumps) > 0 else 0

        Z1 = np.random.normal(0, 1, (simulations, days))
        gbm = (mu - 0.5 * sigma ** 2) + sigma * Z1

        if lambda_j > 0:
            N = np.random.poisson(lambda_j, (simulations, days))
            Z2 = np.random.normal(mu_j, sigma_j, (simulations, days))
            gbm += N * Z2

        final_prices = prices[-1] * np.exp(np.cumsum(gbm, axis=1)[:, -1])
        var_95 = np.percentile(final_prices, 5)
        return ((var_95 - prices[-1]) / prices[-1]) * 100
    except:
        return -100.0


def check_technicals(df):
    if len(df) < 30: return 0.0, 0.0, 0.0, False, False, []
    try:
        prices = df['p'].values;
        volumes = df['v'].values
        sparkline = prices[-15:].tolist()

        y = np.log(prices);
        x = np.arange(len(y))
        r_sq = stats.linregress(x, y).rvalue ** 2

        avg_30 = np.mean(prices[-30:])
        discount_pct = ((prices[-1] - avg_30) / avg_30) * 100

        avg_vol = np.mean(volumes[-30:])
        max_vol_spike = np.max(volumes[-5:]) / (avg_vol if avg_vol > 0 else 1)

        is_golden = False
        if len(prices) >= 200:
            if np.mean(prices[-50:]) > np.mean(prices[-200:]) and prices[-1] > np.mean(prices[-50:]): is_golden = True

        s = pd.Series(prices)
        macd = s.ewm(span=12).mean() - s.ewm(span=26).mean()
        sig = macd.ewm(span=9).mean()
        is_macd = True if macd.iloc[-1] > sig.iloc[-1] else False

        return r_sq, discount_pct, max_vol_spike, is_golden, is_macd, sparkline
    except:
        return 0.0, 0.0, 0.0, False, False, []


def check_news_sentiment(company_name):
    if company_name in st.session_state['news_cache']: return st.session_state['news_cache'][company_name]
    try:
        time.sleep(0.3)
        query = urllib.parse.quote(f"{company_name} Sri Lanka")
        feed = feedparser.parse(f"https://news.google.com/rss/search?q={query}&hl=en-LK&gl=LK&ceid=LK:en")

        if not feed.entries: return 0.0, "⚪ No recent news.", ""

        total = 0;
        count = 0;
        headlines = []
        for entry in feed.entries[:3]:
            total += TextBlob(entry.title).sentiment.polarity;
            count += 1
            headlines.append(f"• {entry.title}")

        avg = total / count if count else 0
        news = "\n".join(headlines)

        res = (0.0, "⚪ Neutral", news)
        if avg < -0.05:
            res = (-1.0, "🔴 Negative", news)
        elif avg > 0.05:
            res = (1.0, "🟢 Positive", news)

        st.session_state['news_cache'][company_name] = res
        return res
    except:
        return 0.0, "⚪ Error", ""


def analyze_financial_reports(company_name, fallback_news=""):
    """
    Tries to find specific numbers. If fails, uses the fallback headline.
    """
    if company_name in st.session_state['fin_cache']: return st.session_state['fin_cache'][company_name]
    try:
        time.sleep(0.5)
        query = urllib.parse.quote(f"{company_name} Sri Lanka business news")
        feed = feedparser.parse(f"https://news.google.com/rss/search?q={query}&hl=en-LK&gl=LK&ceid=LK:en")
        highlights = []
        fin_keywords = ['profit', 'loss', 'revenue', 'income', 'dividend', 'growth', 'earnings', 'debt', 'billion',
                        'million', 'Rs.', 'LKR']

        for entry in feed.entries[:5]:
            title = entry.title;
            lower_title = title.lower()
            has_number = any(c.isdigit() for c in title)
            has_finance_word = any(k in lower_title for k in fin_keywords)
            if has_number and has_finance_word:
                clean = title.split("-")[0].strip()
                highlights.append(f"🔹 {clean}")

        # LOGIC: If no specific 'financial' headline found, fallback to the General News headline
        # This prevents the "No Data" issue when news IS present but just not 'numbered'.
        if not highlights:
            if fallback_news and "No recent" not in fallback_news:
                # Use the top headline from general news as a proxy
                clean_fallback = fallback_news.split("\n")[0].replace("• ", "🔹 ")
                res = f"{clean_fallback} (Gen. News)"
            else:
                res = "No specific data."
        else:
            res = "\n".join(highlights[:2])

        st.session_state['fin_cache'][company_name] = res
        return res
    except:
        return "Analysis Failed"


# --- UI ---
api = CSEBackend()

with st.sidebar:
    st.header("📂 My Portfolio")
    with st.form("add_stock"):
        c1, c2 = st.columns(2)
        new_stock = c1.text_input("Code").upper().strip()
        new_price = c2.number_input("Price", 0.0)
        new_qty = st.number_input("Qty", 1)
        if st.form_submit_button("Add") and new_stock:
            add_to_portfolio(new_stock, new_price, new_qty);
            st.success("Added");
            st.rerun()
    if st.button("🗑️ Clear"): clear_portfolio(); st.rerun()
    st.info("💡 **Signals:**\n✂️ Cut Loss (-10%)\n💰 Profit (+20%)\n🚨 News Alert")

st.title("🛡️ CSE Command Center (v55 Transparent)")

df_all = api.get_all_stocks()
display_df = pd.DataFrame()
if not df_all.empty:
    cols = ['symbol', 'name', 'price', 'sharevolume', 'sectorName']
    available_cols = [c for c in cols if c in df_all.columns]
    display_df = df_all[available_cols].copy()
    display_df.rename(columns={'symbol': 'Code', 'name': 'Company', 'price': 'Price', 'sharevolume': 'Volume',
                               'sectorName': 'Sector'}, inplace=True)
    display_df['Price'] = pd.to_numeric(display_df['Price'], errors='coerce')
    display_df['Volume'] = pd.to_numeric(display_df['Volume'], errors='coerce')
    if 'Sector' not in display_df: display_df['Sector'] = "Unknown"
    display_df = display_df[~display_df['Sector'].str.contains("Debenture|Unit Trust", case=False, na=False)]

aspi, aspi_chg, _, _, turnover = api.get_market_summary()
if aspi_chg < -2.0:
    st.error(f"🚨 MARKET CRASH WARNING: ASPI is down {aspi_chg}%. Buying is NOT recommended today.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("ASPI", f"{aspi:,.2f}", f"{aspi_chg:+.2f}%")
c3.metric("Turnover", f"LKR {turnover:,.2f} M")

my_portfolio = load_portfolio()
if not my_portfolio.empty and not display_df.empty:
    st.subheader("📂 Portfolio Health")
    health = check_portfolio_health(my_portfolio, display_df)
    st.dataframe(health, column_config={"Trend": st.column_config.LineChartColumn("15-Day Trend", y_min=0),
                                        "Profit/Loss %": st.column_config.NumberColumn(format="%.2f%%")},
                 use_container_width=True)

st.divider()
st.subheader("🤖 Master Strategy Scanner")
c1, c2, c3 = st.columns(3)
p_min = c1.slider("Min Price", 0, 100, 2)
v_min = c2.number_input("Min Volume", value=1000)
w_sens = c3.slider("Whale Sensitivity", 1.5, 5.0, 2.5)

if st.button("🚀 RUN MASTER SCAN", type="primary"):
    st.session_state['news_cache'] = {}
    st.session_state['fin_cache'] = {}
    targets = display_df[(display_df['Volume'] >= v_min) & (display_df['Price'] >= p_min)]['Code'].tolist()

    raw_candidates = []
    bar = st.progress(0);
    txt = st.empty()

    # PHASE 1: MATH SCAN
    for i, code in enumerate(targets):
        if i % 5 == 0: txt.text(f"Scanning Math {code} ({i}/{len(targets)})...")
        _, raw = api.get_chart_data(code, 5)
        if raw:
            df = pd.DataFrame(raw)
            df['p'] = pd.to_numeric(df['p'], errors='coerce')
            df['v'] = pd.to_numeric(df.get('v', 0), errors='coerce')
            df.dropna(subset=['p'], inplace=True)

            sim = run_heavy_monte_carlo(df)
            risk = run_jump_stress_test(df)
            r2, disc, whale, gold, macd, spark = check_technicals(df)

            if sim and risk is not None:
                cp = abs(risk) if risk < 0 else 0

                # Math Scores
                scores = {
                    "Momentum": (sim['Upside'] * r2) - (cp * 0.5),
                    "Value": (abs(disc) * 2) - cp if disc < -2 else -100,
                    "Stable": (r2 * 100) - (abs(risk) * 2) if sim['Upside'] > 0 and r2 > 0.5 else -100,
                    "Whale": whale * 10 if whale > w_sens and risk > -15 else -100,
                    "Golden": 100 + sim['Upside'] if gold and sim['Upside'] > 0 else -100,
                    "MACD": 100 + sim['Upside'] if macd and sim['Upside'] > 0 and r2 > 0.3 else -100
                }

                if any(v > 0 for v in scores.values()):
                    try:
                        info = display_df[display_df['Code'] == code].iloc[0]; name = info['Company']; price = info[
                            'Price']
                    except:
                        name = code; price = 0

                    raw_candidates.append({
                        "Stock": code, "Company": name, "Price": price,
                        "Upside%": sim['Upside'] / 100, "Risk%": risk / 100, "R2": r2, "Vol(x)": whale, "Trend": spark,
                        "MathScores": scores, "TotalMathScore": sum(v for v in scores.values() if v > 0)
                    })
        bar.progress((i + 1) / len(targets))

    # PHASE 2: NEWS SCAN & FINAL RANKING
    if raw_candidates:
        # Increase scan limit to top 60 to prevent missing edge cases
        top_candidates = sorted(raw_candidates, key=lambda x: x['TotalMathScore'], reverse=True)[:60]

        final_results = []
        rejected_count = 0
        txt.text(f"Analyzing News for Top {len(top_candidates)} Candidates...")

        for idx, item in enumerate(top_candidates):
            bar.progress((idx + 1) / len(top_candidates))

            # Analyze News
            s_score, s_summ, headlines_txt = check_news_sentiment(item['Company'])

            # VETO: Filter out bad news
            if s_score < 0:
                rejected_count += 1
                continue

                # Financial Analysis (With Fallback)
            fin_report = analyze_financial_reports(item['Company'], fallback_news=headlines_txt)

            scores = item['MathScores']
            badges = []
            strat_count = 0

            if scores["Momentum"] > 0: badges.append("🚀"); strat_count += 1
            if scores["Value"] > 0: badges.append("💎"); strat_count += 1
            if scores["Stable"] > 0: badges.append("🛡️"); strat_count += 1
            if scores["Whale"] > 0: badges.append("🐋"); strat_count += 1
            if scores["Golden"] > 0: badges.append("🌟"); strat_count += 1
            if scores["MACD"] > 0: badges.append("⚡"); strat_count += 1

            if s_score > 0:
                badges.append("📰")
                strat_count += 1
                item['TotalMathScore'] += 50

            item['Strategies'] = strat_count
            item['Badges'] = " ".join(badges)
            item['News'] = s_summ
            item['Financials'] = fin_report

            for k, v in scores.items():
                item[f"{k}_S"] = v

            final_results.append(item)

        res_df = pd.DataFrame(final_results)

        # SUMMARY BAR
        st.info(
            f"📊 **Scan Report:** Scanned {len(targets)} stocks $\\rightarrow$ {len(raw_candidates)} passed Math $\\rightarrow$ **{rejected_count} removed due to Bad News**.")

        # --- MASTER TABLE ---
        if not res_df.empty:
            master_df = res_df.sort_values(by=['Strategies', 'TotalMathScore'], ascending=[False, False]).head(10)

            st.subheader("🏆 Ultimate Top 10 (7-Factor Confluence)")

            col_cfg = {
                "Trend": st.column_config.LineChartColumn("Trend (15d)"),
                "Price": st.column_config.NumberColumn(format="%.2f"),
                "Upside%": st.column_config.NumberColumn(format="%.2f%%"),
                "Strategies": st.column_config.ProgressColumn("Signal Strength", min_value=0, max_value=7,
                                                              format="%d/7"),
                "Financials": st.column_config.TextColumn(width="medium")
            }

            st.dataframe(
                master_df[
                    ['Stock', 'Company', 'Price', 'Strategies', 'Badges', 'Upside%', 'Trend', 'News', 'Financials']],
                column_config=col_cfg, use_container_width=True
            )

            st.divider()
            st.subheader("🔎 Strategy Breakdown")
            tabs = st.tabs(["🚀 Momentum", "💎 Value", "🛡️ Stable", "🐋 Whale", "🌟 Golden", "🚀 MACD"])

            col_cfg_sub = {
                "Trend": st.column_config.LineChartColumn("Trend"),
                "Price": st.column_config.NumberColumn(format="%.2f"),
                "Upside%": st.column_config.NumberColumn(format="%.2f%%")
            }

            strat_cols = ["Momentum_S", "Value_S", "Stable_S", "Whale_S", "Golden_S", "MACD_S"]

            for i, strat in enumerate(strat_cols):
                with tabs[i]:
                    sub_df = res_df[res_df[strat] > 0].sort_values(by=strat, ascending=False).head(5)
                    if not sub_df.empty:
                        st.dataframe(sub_df[['Stock', 'Company', 'Price', 'Upside%', 'Trend', 'News']],
                                     column_config=col_cfg_sub, use_container_width=True)
                    else:
                        st.info("No candidates for this specific strategy in the shortlist.")

            st.success("Master Analysis Complete!")
        else:
            st.warning("All candidates were filtered out by the News Veto.")
    else:
        st.error("No valid market data found.")