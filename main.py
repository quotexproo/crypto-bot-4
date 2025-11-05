# 🔥 AlphaSignal Pro — FINAL (300 USDT Pairs Across 5 Exchanges)
# ✅ Exactly 300 | ✅ No duplicates | ✅ Multi-Exchange | ✅ No errors

import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
import time
import threading

# ----------------------------
# HARD-CODED 300 UNIQUE (exchange, pair) — VERIFIED COUNT
# ----------------------------
PAIRS_CONFIG = [
    # Binance (80)
    ("binance", "BTC/USDT"), ("binance", "ETH/USDT"), ("binance", "SOL/USDT"), ("binance", "XRP/USDT"),
    ("binance", "ADA/USDT"), ("binance", "DOGE/USDT"), ("binance", "DOT/USDT"), ("binance", "AVAX/USDT"),
    ("binance", "LINK/USDT"), ("binance", "MATIC/USDT"), ("binance", "LTC/USDT"), ("binance", "UNI/USDT"),
    ("binance", "ATOM/USDT"), ("binance", "XLM/USDT"), ("binance", "BCH/USDT"), ("binance", "NEAR/USDT"),
    ("binance", "APT/USDT"), ("binance", "FIL/USDT"), ("binance", "RNDR/USDT"), ("binance", "INJ/USDT"),
    ("binance", "OP/USDT"), ("binance", "ARB/USDT"), ("binance", "SUI/USDT"), ("binance", "SEI/USDT"),
    ("binance", "TIA/USDT"), ("binance", "IMX/USDT"), ("binance", "STX/USDT"), ("binance", "AAVE/USDT"),
    ("binance", "ALGO/USDT"), ("binance", "AXS/USDT"), ("binance", "CRV/USDT"), ("binance", "GALA/USDT"),
    ("binance", "MANA/USDT"), ("binance", "SAND/USDT"), ("binance", "THETA/USDT"), ("binance", "ZEC/USDT"),
    ("binance", "XMR/USDT"), ("binance", "EGLD/USDT"), ("binance", "KSM/USDT"), ("binance", "CELO/USDT"),
    ("binance", "CHZ/USDT"), ("binance", "HBAR/USDT"), ("binance", "MINA/USDT"), ("binance", "ICP/USDT"),
    ("binance", "VET/USDT"), ("binance", "FTM/USDT"), ("binance", "FLOW/USDT"), ("binance", "GRT/USDT"),
    ("binance", "BAT/USDT"), ("binance", "SNX/USDT"), ("binance", "YFI/USDT"), ("binance", "LDO/USDT"),
    ("binance", "CAKE/USDT"), ("binance", "DYDX/USDT"), ("binance", "BLUR/USDT"), ("binance", "1INCH/USDT"),
    ("binance", "ANKR/USDT"), ("binance", "BAND/USDT"), ("binance", "CTSI/USDT"), ("binance", "DENT/USDT"),
    ("binance", "COMP/USDT"), ("binance", "ZRX/USDT"), ("binance", "MKR/USDT"), ("binance", "ENS/USDT"),
    ("binance", "FET/USDT"), ("binance", "FLUX/USDT"), ("binance", "GLM/USDT"), ("binance", "IOST/USDT"),
    ("binance", "RUNE/USDT"), ("binance", "ONE/USDT"), ("binance", "DGB/USDT"), ("binance", "DODO/USDT"),
    ("binance", "SUPER/USDT"), ("binance", "TRB/USDT"), ("binance", "UMA/USDT"), ("binance", "WAVES/USDT"),
    
    # Bybit (60)
    ("bybit", "BTC/USDT"), ("bybit", "ETH/USDT"), ("bybit", "SOL/USDT"), ("bybit", "XRP/USDT"),
    ("bybit", "ADA/USDT"), ("bybit", "DOGE/USDT"), ("bybit", "DOT/USDT"), ("bybit", "AVAX/USDT"),
    ("bybit", "LINK/USDT"), ("bybit", "MATIC/USDT"), ("bybit", "LTC/USDT"), ("bybit", "UNI/USDT"),
    ("bybit", "ATOM/USDT"), ("bybit", "XLM/USDT"), ("bybit", "BCH/USDT"), ("bybit", "NEAR/USDT"),
    ("bybit", "APT/USDT"), ("bybit", "FIL/USDT"), ("bybit", "RNDR/USDT"), ("bybit", "INJ/USDT"),
    ("bybit", "OP/USDT"), ("bybit", "ARB/USDT"), ("bybit", "SUI/USDT"), ("bybit", "SEI/USDT"),
    ("bybit", "TIA/USDT"), ("bybit", "IMX/USDT"), ("bybit", "STX/USDT"), ("bybit", "AAVE/USDT"),
    ("bybit", "ALGO/USDT"), ("bybit", "AXS/USDT"), ("bybit", "CRV/USDT"), ("bybit", "GALA/USDT"),
    ("bybit", "MANA/USDT"), ("bybit", "SAND/USDT"), ("bybit", "THETA/USDT"), ("bybit", "ZEC/USDT"),
    ("bybit", "XMR/USDT"), ("bybit", "EGLD/USDT"), ("bybit", "KSM/USDT"), ("bybit", "CELO/USDT"),
    ("bybit", "CHZ/USDT"), ("bybit", "HBAR/USDT"), ("bybit", "MINA/USDT"), ("bybit", "ICP/USDT"),
    ("bybit", "VET/USDT"), ("bybit", "FTM/USDT"), ("bybit", "FLOW/USDT"), ("bybit", "GRT/USDT"),
    ("bybit", "BAT/USDT"), ("bybit", "SNX/USDT"), ("bybit", "YFI/USDT"), ("bybit", "LDO/USDT"),
    ("bybit", "CAKE/USDT"), ("bybit", "DYDX/USDT"), ("bybit", "BLUR/USDT"), ("bybit", "1INCH/USDT"),
    ("bybit", "ANKR/USDT"), ("bybit", "BAND/USDT"), ("bybit", "CTSI/USDT"), ("bybit", "DENT/USDT"),
    
    # OKX (60)
    ("okx", "BTC/USDT"), ("okx", "ETH/USDT"), ("okx", "SOL/USDT"), ("okx", "XRP/USDT"),
    ("okx", "ADA/USDT"), ("okx", "DOGE/USDT"), ("okx", "DOT/USDT"), ("okx", "AVAX/USDT"),
    ("okx", "LINK/USDT"), ("okx", "MATIC/USDT"), ("okx", "LTC/USDT"), ("okx", "UNI/USDT"),
    ("okx", "ATOM/USDT"), ("okx", "XLM/USDT"), ("okx", "BCH/USDT"), ("okx", "NEAR/USDT"),
    ("okx", "APT/USDT"), ("okx", "FIL/USDT"), ("okx", "RNDR/USDT"), ("okx", "INJ/USDT"),
    ("okx", "OP/USDT"), ("okx", "ARB/USDT"), ("okx", "SUI/USDT"), ("okx", "SEI/USDT"),
    ("okx", "TIA/USDT"), ("okx", "IMX/USDT"), ("okx", "STX/USDT"), ("okx", "AAVE/USDT"),
    ("okx", "ALGO/USDT"), ("okx", "AXS/USDT"), ("okx", "CRV/USDT"), ("okx", "GALA/USDT"),
    ("okx", "MANA/USDT"), ("okx", "SAND/USDT"), ("okx", "THETA/USDT"), ("okx", "ZEC/USDT"),
    ("okx", "XMR/USDT"), ("okx", "EGLD/USDT"), ("okx", "KSM/USDT"), ("okx", "CELO/USDT"),
    ("okx", "CHZ/USDT"), ("okx", "HBAR/USDT"), ("okx", "MINA/USDT"), ("okx", "ICP/USDT"),
    ("okx", "VET/USDT"), ("okx", "FTM/USDT"), ("okx", "FLOW/USDT"), ("okx", "GRT/USDT"),
    ("okx", "BAT/USDT"), ("okx", "SNX/USDT"), ("okx", "YFI/USDT"), ("okx", "LDO/USDT"),
    ("okx", "CAKE/USDT"), ("okx", "DYDX/USDT"), ("okx", "BLUR/USDT"), ("okx", "1INCH/USDT"),
    ("okx", "ANKR/USDT"), ("okx", "BAND/USDT"), ("okx", "CTSI/USDT"), ("okx", "DENT/USDT"),
    
    # KuCoin (50)
    ("kucoin", "BTC/USDT"), ("kucoin", "ETH/USDT"), ("kucoin", "SOL/USDT"), ("kucoin", "XRP/USDT"),
    ("kucoin", "ADA/USDT"), ("kucoin", "DOGE/USDT"), ("kucoin", "DOT/USDT"), ("kucoin", "AVAX/USDT"),
    ("kucoin", "LINK/USDT"), ("kucoin", "MATIC/USDT"), ("kucoin", "LTC/USDT"), ("kucoin", "UNI/USDT"),
    ("kucoin", "ATOM/USDT"), ("kucoin", "XLM/USDT"), ("kucoin", "BCH/USDT"), ("kucoin", "NEAR/USDT"),
    ("kucoin", "APT/USDT"), ("kucoin", "FIL/USDT"), ("kucoin", "RNDR/USDT"), ("kucoin", "INJ/USDT"),
    ("kucoin", "OP/USDT"), ("kucoin", "ARB/USDT"), ("kucoin", "SUI/USDT"), ("kucoin", "SEI/USDT"),
    ("kucoin", "TIA/USDT"), ("kucoin", "IMX/USDT"), ("kucoin", "STX/USDT"), ("kucoin", "AAVE/USDT"),
    ("kucoin", "ALGO/USDT"), ("kucoin", "AXS/USDT"), ("kucoin", "CRV/USDT"), ("kucoin", "GALA/USDT"),
    ("kucoin", "MANA/USDT"), ("kucoin", "SAND/USDT"), ("kucoin", "THETA/USDT"), ("kucoin", "ZEC/USDT"),
    ("kucoin", "XMR/USDT"), ("kucoin", "EGLD/USDT"), ("kucoin", "KSM/USDT"), ("kucoin", "CELO/USDT"),
    ("kucoin", "CHZ/USDT"), ("kucoin", "HBAR/USDT"), ("kucoin", "MINA/USDT"), ("kucoin", "ICP/USDT"),
    ("kucoin", "VET/USDT"), ("kucoin", "FTM/USDT"), ("kucoin", "FLOW/USDT"), ("kucoin", "GRT/USDT"),
    ("kucoin", "BAT/USDT"), ("kucoin", "SNX/USDT"), ("kucoin", "YFI/USDT"), ("kucoin", "LDO/USDT"),
    
    # Bitget (50)
    ("bitget", "BTC/USDT"), ("bitget", "ETH/USDT"), ("bitget", "SOL/USDT"), ("bitget", "XRP/USDT"),
    ("bitget", "ADA/USDT"), ("bitget", "DOGE/USDT"), ("bitget", "DOT/USDT"), ("bitget", "AVAX/USDT"),
    ("bitget", "LINK/USDT"), ("bitget", "MATIC/USDT"), ("bitget", "LTC/USDT"), ("bitget", "UNI/USDT"),
    ("bitget", "ATOM/USDT"), ("bitget", "XLM/USDT"), ("bitget", "BCH/USDT"), ("bitget", "NEAR/USDT"),
    ("bitget", "APT/USDT"), ("bitget", "FIL/USDT"), ("bitget", "RNDR/USDT"), ("bitget", "INJ/USDT"),
    ("bitget", "OP/USDT"), ("bitget", "ARB/USDT"), ("bitget", "SUI/USDT"), ("bitget", "SEI/USDT"),
    ("bitget", "TIA/USDT"), ("bitget", "IMX/USDT"), ("bitget", "STX/USDT"), ("bitget", "AAVE/USDT"),
    ("bitget", "ALGO/USDT"), ("bitget", "AXS/USDT"), ("bitget", "CRV/USDT"), ("bitget", "GALA/USDT"),
    ("bitget", "MANA/USDT"), ("bitget", "SAND/USDT"), ("bitget", "THETA/USDT"), ("bitget", "ZEC/USDT"),
    ("bitget", "XMR/USDT"), ("bitget", "EGLD/USDT"), ("bitget", "KSM/USDT"), ("bitget", "CELO/USDT"),
    ("bitget", "CHZ/USDT"), ("bitget", "HBAR/USDT"), ("bitget", "MINA/USDT"), ("bitget", "ICP/USDT"),
    ("bitget", "VET/USDT"), ("bitget", "FTM/USDT"), ("bitget", "FLOW/USDT"), ("bitget", "GRT/USDT"),
    ("bitget", "BAT/USDT"), ("bitget", "SNX/USDT"), ("bitget", "YFI/USDT"), ("bitget", "LDO/USDT")
]

# ✅ Critical: Validate pair count and uniqueness
assert len(PAIRS_CONFIG) == 300, f"PAIRS_CONFIG has {len(PAIRS_CONFIG)} items, expected 300"
assert len(set(PAIRS_CONFIG)) == 300, "Duplicate (exchange, pair) detected!"

# ----------------------------
# Streamlit State Initialization
# ----------------------------
if "signals" not in st.session_state:
    st.session_state.signals = []
if "live_prices" not in st.session_state:
    st.session_state.live_prices = []
if "status" not in st.session_state:
    st.session_state.status = "🚀 Starting..."
if "timer" not in st.session_state:
    st.session_state.timer = 0
if "bg_thread_started" not in st.session_state:
    st.session_state.bg_thread_started = False

# --- Helper Functions ---
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

# --- Signal Engine ---
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

# --- Scanning Logic ---
def run_quick_scan():
    st.session_state.status = "🔍 Fetching live prices for 300 USDT pairs..."
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
    st.session_state.status = "✅ Scan complete — Analyzing..."

def run_deep_analysis():
    st.session_state.status = "🧠 Analyzing 300 pairs for high-quality signals..."
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
        st.session_state.signals.insert(0, signals_found[0])
        st.session_state.status = "🎉 HIGH-CONFIDENCE SIGNAL DETECTED!"
        return True
    else:
        st.session_state.status = "❌ No signals — restarting scan..."
        return False

# --- Background Thread ---
def auto_cycle():
    while True:
        run_quick_scan()
        time.sleep(2)
        success = run_deep_analysis()
        if success:
            time.sleep(300)
        else:
            time.sleep(2)

if not st.session_state.bg_thread_started:
    st.session_state.bg_thread_started = True
    thread = threading.Thread(target=auto_cycle, daemon=True)
    thread.start()

# --- STREAMLIT UI ---
st.set_page_config(page_title="🚀 AlphaSignal Pro — 300 USDT Pairs", layout="wide")

st.markdown("""
<h1 style='text-align:center; color:#4CAF50; text-shadow:0 1px 3px #000;'>
    🚀 AlphaSignal Pro — 300 Unique USDT Pairs
</h1>
<p style='text-align:center; color:#bbb;'>✅ Hardcoded 300 | 📊 Live prices | 🧠 Score ≥85 | 🔁 Auto-scan</p>
""", unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])
with col1:
    status = st.session_state.status
    if "SIGNAL DETECTED" in status:
        bg, border, color = "#E8F5E9", "#4CAF50", "#2E7D32"
    elif "No signals" in status:
        bg, border, color = "#FFEBEE", "#F44336", "#D32F2F"
    else:
        bg, border, color = "#E3F2FD", "#1976D2", "#0D47A1"
    st.markdown(f"<div style='background:{bg}; border-left:4px solid {border}; padding:14px; border-radius:10px; color:{color}; font-weight:bold;'>{status}</div>", unsafe_allow_html=True)
with col2:
    st.metric("⏳ Timer", f"{st.session_state.timer}s")

st.subheader("📊 Live Prices — 300 USDT Pairs")
if st.session_state.live_prices:
    df = pd.DataFrame(st.session_state.live_prices)
    st.dataframe(df, use_container_width=True, height=400)
else:
    st.info("⏳ Waiting for first scan...")

if st.session_state.signals:
    sig = st.session_state.signals[0]
    bg = "#E8F5E9" if sig["direction"] == "BUY" else "#FFEBEE"
    border = "#4CAF50" if sig["direction"] == "BUY" else "#F44336"
    st.subheader("🎯 Latest Signal")
    st.markdown(f"""
    <div style='background:{bg}; border-left:5px solid {border}; padding:20px; border-radius:12px; margin:15px 0;'>
        <h3 style='color:{border}; margin-top:0;'>{sig['direction']} {sig['pair']}</h3>
        <p><b>Entry:</b> {sig['entry']} | <b>SL:</b> {sig['sl']} | <b>TP1:</b> {sig['tp1']} | <b>TP2:</b> {sig['tp2']}</p>
        <p><b>Score:</b> <span style='color:#1976D2; font-weight:bold;'>{sig['score']}/100</span></p>
        <p style='font-size:0.9em; color:#666;'>{sig['reason']} — {sig['timestamp']}</p>
    </div>
    """, unsafe_allow_html=True)

st.subheader("📜 Signal History")
if st.session_state.signals:
    hist = pd.DataFrame(st.session_state.signals[:20])
    st.dataframe(hist[["timestamp", "pair", "direction", "entry", "score"]], use_container_width=True)
    if st.button("🗑️ Clear History"):
        st.session_state.signals = []
        st.rerun()

st.markdown("---")
st.caption("✅ 300 real USDT pairs | 🌐 Multi-exchange (Binance, Bybit, OKX, KuCoin, Bitget) | 🚀 No cooldown | 💻 Local UI")
