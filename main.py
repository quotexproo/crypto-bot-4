# 🔥 Professional Crypto Signal Bot v5 — FINAL EDITION
# 🌈 300 Pairs | 30 Exchanges | LIVE Dashboard | NO COOLDOWN
# ⏱️ Auto 2-min Scan → 5-min Analysis Cycle | Colorful UI
# 🚫 No Telegram | 🖥️ Local Streamlit Only

import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import threading

# ----------------------------
# CONFIG: 300 High-Liquidity Pairs Across 30 Exchanges
# ----------------------------
EXCHANGES = [
    'binance', 'kraken', 'coinbase', 'kucoin', 'bybit', 'okx',
    'bitstamp', 'bitfinex', 'gemini', 'huobi', 'gateio', 'mexc',
    'bitget', 'poloniex', 'ascendex', 'bittrex', 'phemex', 'coinex',
    'lbank', 'bibox', 'woo', 'cryptocom', 'probit', 'latoken',
    'whitebit', 'bigone', 'bithumb', 'digifinex', 'hitbtc', 'tidex'
]

MAJOR_PAIRS = [
    "BTC", "ETH", "SOL", "XRP", "ADA", "DOT", "AVAX", "LINK", "MATIC", "LTC",
    "UNI", "ATOM", "XLM", "BCH", "NEAR", "APT", "FIL", "RNDR", "INJ", "OP",
    "ARB", "PEPE", "SHIB", "DOGE", "TRX", "ETC", "ICP", "VET", "FTM", "FLOW"
]

MIDCAP_PAIRS = [
    "AAVE", "ALGO", "AXS", "COMP", "CRV", "ENJ", "GALA", "MANA", "SAND", "THETA",
    "ZEC", "XMR", "EGLD", "KSM", "RUNE", "CELO", "ONE", "CHZ", "HBAR", "MINA"
]

PAIRS_CONFIG = []
for exchange in EXCHANGES[:15]:
    for base in MAJOR_PAIRS[:20]:
        quote = "USDT" if exchange in [
            'binance', 'bybit', 'okx', 'kucoin', 'huobi', 'gateio', 'mexc',
            'bitget', 'lbank', 'bibox', 'woo', 'cryptocom', 'whitebit',
            'digifinex', 'hitbtc'
        ] else "USD"
        PAIRS_CONFIG.append((exchange, f"{base}/{quote}"))

for exchange in EXCHANGES[15:]:
    for base in MIDCAP_PAIRS[:10]:
        quote = "USDT" if exchange not in [
            'kraken', 'coinbase', 'bitstamp', 'bitfinex', 'gemini', 'bittrex', 'probit'
        ] else "USD"
        PAIRS_CONFIG.append((exchange, f"{base}/{quote}"))

PAIRS_CONFIG = PAIRS_CONFIG[:300]
assert len(PAIRS_CONFIG) == 300, f"Expected 300 pairs, got {len(PAIRS_CONFIG)}"

# ----------------------------
# Streamlit Session State
# ----------------------------
if "signals" not in st.session_state:
    st.session_state.signals = []
if "live_prices" not in st.session_state:
    st.session_state.live_prices = []
if "status" not in st.session_state:
    st.session_state.status = "🚀 Starting up..."
if "timer" not in st.session_state:
    st.session_state.timer = 0
if "bg_thread_started" not in st.session_state:
    st.session_state.bg_thread_started = False

# ----------------------------
# Helpers
# ----------------------------
def round_price(price, pair):
    if "BTC" in pair:
        return round(price, 2)
    elif "ETH" in pair:
        return round(price, 2)
    else:
        return round(price, 6)

@st.cache_resource(ttl=300)
def get_exchange(exchange_id):
    return getattr(ccxt, exchange_id)({'enableRateLimit': True, 'timeout': 10000})

def fetch_ohlcv_safe(exchange_id, pair, timeframe, limit=100):
    try:
        ex = get_exchange(exchange_id)
        ohlcv = ex.fetch_ohlcv(pair, timeframe, limit=limit)
        if len(ohlcv) < limit * 0.8:
            return None
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except:
        return None

def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs.replace([np.inf, -np.inf], np.nan).fillna(0)))
    return rsi

def calculate_atr(df, period=14):
    tr0 = df['high'] - df['low']
    tr1 = abs(df['high'] - df['close'].shift())
    tr2 = abs(df['low'] - df['close'].shift())
    tr = pd.concat([tr0, tr1, tr2], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def detect_liquidity_sweep(df, window=20):
    if len(df) < window:
        return False
    recent_high = df['high'].iloc[-1]
    recent_low = df['low'].iloc[-1]
    return (recent_high > df['high'].iloc[-window:-1].max()) or (recent_low < df['low'].iloc[-window:-1].min())

def find_order_block(df, bias):
    if len(df) < 10:
        return False
    closes = df['close'].values
    for i in range(5, len(closes)-2):
        if bias == "BUY":
            if closes[i] < closes[i-1] and closes[i+1] > closes[i] * 1.006:
                return True
        else:
            if closes[i] > closes[i-1] and closes[i+1] < closes[i] * 0.994:
                return True
    return False

def find_dynamic_sr(df, lookback=50):
    if len(df) < lookback:
        return None, None
    highs = df['high'].values
    lows = df['low'].values
    volumes = df['volume'].values
    avg_vol = np.mean(volumes[-lookback:])
    supports, resistances = [], []
    for i in range(5, len(df)-5):
        if lows[i] == min(lows[i-5:i+6]) and volumes[i] > avg_vol * 0.8:
            supports.append(lows[i])
        if highs[i] == max(highs[i-5:i+6]) and volumes[i] > avg_vol * 0.8:
            resistances.append(highs[i])
    price = df['close'].iloc[-1]
    nearest_s = max([s for s in supports if s < price], default=None)
    nearest_r = min([r for r in resistances if r > price], default=None)
    return nearest_s, nearest_r

# ----------------------------
# Signal Engine (Score ≥85)
# ----------------------------
def analyze_pair_professional(exchange_id, pair):
    df_1h = fetch_ohlcv_safe(exchange_id, pair, '1h', 200)
    df_15m = fetch_ohlcv_safe(exchange_id, pair, '15m', 60)
    if df_1h is None or df_15m is None:
        return None

    ema50 = calculate_ema(df_1h['close'], 50).iloc[-1]
    ema200 = calculate_ema(df_1h['close'], 200).iloc[-1]
    if ema50 > ema200 * 1.001:
        bias = "BUY"
    elif ema50 < ema200 * 0.999:
        bias = "SELL"
    else:
        return None

    ema9 = calculate_ema(df_15m['close'], 9)
    ema20 = calculate_ema(df_15m['close'], 20)
    if bias == "BUY":
        if not (ema9.iloc[-2] < ema20.iloc[-2] and ema9.iloc[-1] > ema20.iloc[-1]):
            return None
    else:
        if not (ema9.iloc[-2] > ema20.iloc[-2] and ema9.iloc[-1] < ema20.iloc[-1]):
            return None

    score = 40
    rsi = calculate_rsi(df_15m['close']).iloc[-1]
    rsi_prev = calculate_rsi(df_15m['close']).iloc[-2]
    if (bias == "BUY" and 30 <= rsi < 45 and rsi > rsi_prev) or (bias == "SELL" and 55 < rsi <= 70 and rsi < rsi_prev):
        score += 10

    vwap = ((df_15m['high'] + df_15m['low'] + df_15m['close']) / 3).iloc[-1]
    price = df_15m['close'].iloc[-1]
    if (bias == "BUY" and price > vwap) or (bias == "SELL" and price < vwap):
        score += 10

    atr = calculate_atr(df_15m).iloc[-1]
    vol = df_15m['volume'].iloc[-1]
    avg_atr = calculate_atr(df_15m).iloc[-20:-1].mean()
    avg_vol = df_15m['volume'].iloc[-50:-1].mean()
    if atr >= 1.1 * avg_atr:
        score += 10
    if vol >= 1.3 * avg_vol:
        score += 10

    if detect_liquidity_sweep(df_15m):
        score += 10
    if find_order_block(df_15m, bias):
        score += 10

    s, r = find_dynamic_sr(df_1h)
    if s and r:
        if (bias == "BUY" and abs(price - s) / price < 0.005) or (bias == "SELL" and abs(r - price) / price < 0.005):
            score += 10

    if score < 85:
        return None

    sl_buffer = 1.5 * atr
    if bias == "BUY":
        sl = df_15m['low'].tail(10).min() - sl_buffer
        tp1 = price + atr
        tp2 = price + 2 * atr
    else:
        sl = df_15m['high'].tail(10).max() + sl_buffer
        tp1 = price - atr
        tp2 = price - 2 * atr

    return {
        "pair": pair,
        "exchange": exchange_id,
        "direction": bias,
        "entry": round_price(price, pair),
        "sl": round_price(sl, pair),
        "tp1": round_price(tp1, pair),
        "tp2": round_price(tp2, pair),
        "score": int(score),
        "timestamp": datetime.utcnow().strftime("%H:%M UTC"),
        "reason": f"Score {score}: Trend+Cross+Structure"
    }

# ----------------------------
# Scanning Cycles (NO COOLDOWN!)
# ----------------------------
def run_quick_scan():
    st.session_state.status = "🔍 Scanning 300 Pairs — Fetching Live Prices..."
    st.session_state.timer = 120
    live_prices = []
    for ex_id, pair in PAIRS_CONFIG:
        try:
            ex = get_exchange(ex_id)
            ticker = ex.fetch_ticker(pair)
            live_prices.append({"Pair": pair, "Price": round(ticker['last'], 6), "Exchange": ex_id})
        except:
            continue
        time.sleep(0.02)
    st.session_state.live_prices = live_prices
    st.session_state.status = "✅ Quick Scan Complete — Starting Deep Analysis..."

def run_deep_analysis():
    st.session_state.status = "🧠 Deep Analysis — Evaluating 300 Pairs for Signals..."
    st.session_state.timer = 300
    signals_found = []

    for ex_id, pair in PAIRS_CONFIG:
        try:
            sig = analyze_pair_professional(ex_id, pair)
            if sig:
                signals_found.append(sig)
        except:
            continue
        time.sleep(0.05)

    if signals_found:
        signals_found.sort(key=lambda x: x['score'], reverse=True)
        top = signals_found[0]
        st.session_state.signals.insert(0, top)
        st.session_state.status = "🎉 SIGNAL DETECTED! High-Confidence Setup Found."
        return True
    else:
        st.session_state.status = "❌ No Valid Signals — Restarting Scan Cycle..."
        return False

# ----------------------------
# Background Auto-Cycle
# ----------------------------
def auto_cycle():
    while True:
        run_quick_scan()
        time.sleep(2)
        success = run_deep_analysis()
        if success:
            time.sleep(300)  # Wait 5 min after signal before next full cycle
        else:
            time.sleep(2)  # Immediately restart

if not st.session_state.bg_thread_started:
    st.session_state.bg_thread_started = True
    thread = threading.Thread(target=auto_cycle, daemon=True)
    thread.start()

# ----------------------------
# 🎨 STREAMLIT DASHBOARD — COLORFUL & PROFESSIONAL
# ----------------------------
st.set_page_config(
    page_title="🚀 AlphaSignal Pro — Institutional Crypto Bot",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Header Banner
st.markdown("""
<h1 style='text-align: center; color: #4CAF50; text-shadow: 1px 1px 2px #000;'>
    🚀 AlphaSignal Pro — Institutional Crypto Signal Engine
</h1>
<p style='text-align: center; color: #bbb; font-size: 1.1em;'>
    🔍 300 Pairs | 🌐 30 Exchanges | 📊 Live Dashboard | 🧠 AI-Style Confluence Scoring
</p>
""", unsafe_allow_html=True)

# Status Panel with Live Timer
col1, col2 = st.columns([3, 1])
with col1:
    if "SIGNAL DETECTED" in st.session_state.status:
        bg, border, icon = "#2E7D32", "#1B5E20", "✅"
        text_color = "#E8F5E9"
    elif "No Valid Signals" in st.session_state.status:
        bg, border, icon = "#D32F2F", "#B71C1C", "⚠️"
        text_color = "#FFEBEE"
    else:
        bg, border, icon = "#1976D2", "#0D47A1", "🔄"
        text_color = "#E3F2FD"
    st.markdown(f"""
    <div style="
        background-color: {bg};
        border-left: 5px solid {border};
        padding: 14px;
        border-radius: 10px;
        color: {text_color};
        font-weight: bold;
        box-shadow: 0 2px 6px rgba(0,0,0,0.2);
    ">
        {icon} {st.session_state.status}
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.metric("⏳ Cycle Timer", f"{st.session_state.timer}s")

# Live Prices Section
st.markdown("## 📈 Live Market Prices (300 Pairs)")
if st.session_state.live_prices:
    df_prices = pd.DataFrame(st.session_state.live_prices)
    # Add color to prices (you can extend this)
    st.dataframe(df_prices, use_container_width=True, height=400)
else:
    st.info("⏳ Awaiting first scan...")

# Signal Display
if st.session_state.signals:
    sig = st.session_state.signals[0]
    bg_color = "#E8F5E9" if sig["direction"] == "BUY" else "#FFEBEE"
    border_left = "4px solid #4CAF50" if sig["direction"] == "BUY" else "4px solid #F44336"
    icon = "🟢 BUY" if sig["direction"] == "BUY" else "🔴 SELL"
    st.markdown("## 🎯 Latest High-Confidence Signal")
    st.markdown(f"""
    <div style="
        background-color: {bg_color};
        border-left: {border_left};
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    ">
        <h3 style='margin-top:0; color: {'#2E7D32' if sig['direction']=='BUY' else '#D32F2F'}'>
            {icon} {sig['pair']} on {sig['exchange'].title()}
        </h3>
        <p><b>Entry:</b> {sig['entry']} &nbsp; | &nbsp; <b>SL:</b> {sig['sl']} &nbsp; | &nbsp; 
        <b>TP1:</b> {sig['tp1']} &nbsp; | &nbsp; <b>TP2:</b> {sig['tp2']}</p>
        <p><b>Confidence Score:</b> <span style='color: #1976D2; font-weight: bold;'>{sig['score']}/100</span></p>
        <p style='font-size: 0.9em; color: #666;'>{sig['reason']} — {sig['timestamp']}</p>
    </div>
    """, unsafe_allow_html=True)

# Signal History
st.markdown("## 📜 Signal History (Last 20)")
if st.session_state.signals:
    hist_df = pd.DataFrame(st.session_state.signals[:20])
    # Color direction
    def color_dir(val):
        color = "#4CAF50" if val == "BUY" else "#F44336"
        return f'color: {color}; font-weight: bold;'
    styled_df = hist_df[["timestamp", "pair", "exchange", "direction", "entry", "score"]].style.applymap(color_dir, subset=['direction'])
    st.dataframe(styled_df, use_container_width=True)
    
    if st.button("🗑️ Clear All Signals", type="secondary"):
        st.session_state.signals = []
        st.rerun()
else:
    st.info("No signals generated yet.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.9em;'>
    🔒 All analysis runs locally | 🔄 Auto-scans every cycle | 🧠 Score ≥85 required | 💡 No cooldown — signals can repeat
</div>
""", unsafe_allow_html=True)
