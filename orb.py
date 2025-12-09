import os
import sys
import time
import json
import requests
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time as dtime
from decimal import Decimal, ROUND_HALF_UP

# Reuse the robust login from kotak_login.py logic
# We import the NeoAPI client and dotenv
from dotenv import load_dotenv
from neo_api_client import NeoAPI
import pyotp

# ==========================================
# ⚙️ STRATEGY CONFIGURATION
# ==========================================
SYMBOLS_TO_TRADE = ["NSE:ADANIENT", "NSE:ADANIPORTS", "NSE:APOLLOHOSP", "NSE:ASIANPAINT", "NSE:AXISBANK", "NSE:BAJAJ-AUTO", "NSE:BAJFINANCE", "NSE:BAJAJFINSV", "NSE:BHARTIARTL", "NSE:CIPLA", "NSE:COALINDIA", "NSE:DIVISLAB", "NSE:DRREDDY", "NSE:EICHERMOT", "NSE:ETERNAL", "NSE:GRASIM", "NSE:HCLTECH", "NSE:HDFCBANK", "NSE:HDFCLIFE", "NSE:HEROMOTOCO", "NSE:HINDALCO", "NSE:HINDUNILVR", "NSE:ICICIBANK", "NSE:INDUSINDBK", "NSE:INFY", "NSE:ITC", "NSE:JIOFIN", "NSE:JSWSTEEL", "NSE:KOTAKBANK", "NSE:LT", "NSE:LTIM", "NSE:M&M", "NSE:MARUTI", "NSE:NESTLEIND", "NSE:NTPC", "NSE:ONGC", "NSE:POWERGRID", "NSE:RELIANCE", "NSE:SBILIFE", "NSE:SBIN", "NSE:SHRIRAMFIN", "NSE:SUNPHARMA", "NSE:TCS", "NSE:TATACONSUM", "NSE:TMPV", "NSE:TATASTEEL", "NSE:TECHM", "NSE:TITAN", "NSE:ULTRACEMCO", "NSE:WIPRO"] # Example subset
# Format: "NSE:SYMBOL" or "BSE:SYMBOL"

RISK_PER_TRADE = 500  # INR per trade risk
INTRADAY_LEVERAGE = 4 # Kotak leverage (used for checking margin, though order is MIS)
SL_PERCENT = 0.5      # Fallback if strategy doesn't define dynamic SL
TARGET_PERCENT = 1.5  # Fallback target

# Strategy Params
ORB_START_TIME = dtime(9, 15)
ORB_END_TIME = dtime(10, 30)
ORB_WINDOW_MINUTES = 75

ADX_PERIOD = 14
ADX_THRESHOLD = 50 # Standard trend strength
SUPERTREND_PERIOD = 7
SUPERTREND_MULTIPLIER = 3
RSI_PERIOD = 14
RSI_UPPER = 72
RSI_LOWER = 28

# Alerting
SEND_TELEGRAM = True

# Data Directory (Must match download_history.py)
DATA_DIR = "warmup_data"

# ==========================================
# 📊 INDICATOR FUNCTIONS (Reused)
# ==========================================
def calculate_adx(df, period=14):
    df = df.copy()
    df['tr0'] = abs(df['high'] - df['low'])
    df['tr1'] = abs(df['high'] - df['close'].shift(1))
    df['tr2'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)

    df['pdm'] = df['high'] - df['high'].shift(1)
    df['ndm'] = df['low'].shift(1) - df['low']

    df['pdm'] = df.apply(lambda x: x['pdm'] if (x['pdm'] > x['ndm'] and x['pdm'] > 0) else 0, axis=1)
    df['ndm'] = df.apply(lambda x: x['ndm'] if (x['ndm'] > x['pdm'] and x['ndm'] > 0) else 0, axis=1)

    df['tr_s'] = df['tr'].ewm(alpha=1/period, adjust=False).mean()
    df['pdm_s'] = df['pdm'].ewm(alpha=1/period, adjust=False).mean()
    df['ndm_s'] = df['ndm'].ewm(alpha=1/period, adjust=False).mean()

    df['pdi'] = 100 * (df['pdm_s'] / df['tr_s'])
    df['ndi'] = 100 * (df['ndm_s'] / df['tr_s'])

    df['dx'] = 100 * abs(df['pdi'] - df['ndi']) / (df['pdi'] + df['ndi'])
    df['adx'] = df['dx'].ewm(alpha=1/period, adjust=False).mean()
    return df['adx']

def calculate_supertrend(df, period=7, multiplier=3):
    df = df.copy()
    df['tr0'] = abs(df['high'] - df['low'])
    df['tr1'] = abs(df['high'] - df['close'].shift(1))
    df['tr2'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
    df['atr'] = df['tr'].ewm(alpha=1/period, adjust=False).mean()

    df['hl2'] = (df['high'] + df['low']) / 2
    df['basic_upper'] = df['hl2'] + (multiplier * df['atr'])
    df['basic_lower'] = df['basic_lower'] = df['hl2'] - (multiplier * df['atr'])

    # Vectorized implementation for speed optimization would be better but keeping loop for correctness parity
    close = df['close'].values
    basic_upper = df['basic_upper'].values
    basic_lower = df['basic_lower'].values
    final_upper = np.zeros(len(df))
    final_lower = np.zeros(len(df))
    supertrend = np.zeros(len(df))
    trend = np.zeros(len(df))

    # Initialize first valid value
    final_upper[0] = basic_upper[0]
    final_lower[0] = basic_lower[0]
    trend[0] = 1

    for i in range(1, len(df)):
        if basic_upper[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]: final_upper[i] = basic_upper[i]
        else: final_upper[i] = final_upper[i-1]

        if basic_lower[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]: final_lower[i] = basic_lower[i]
        else: final_lower[i] = final_lower[i-1]

        if trend[i-1] == 1:
            if close[i] <= final_lower[i]: trend[i] = -1; supertrend[i] = final_upper[i]
            else: trend[i] = 1; supertrend[i] = final_lower[i]
        else:
            if close[i] >= final_upper[i]: trend[i] = 1; supertrend[i] = final_lower[i]
            else: trend[i] = -1; supertrend[i] = final_upper[i]

    df['supertrend'] = supertrend
    df['trend'] = trend
    return df

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# ==========================================
# 🛠 UTILS
# ==========================================
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def telegram_alert(msg):
    if not SEND_TELEGRAM: return
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if token and chat_id:
        try:
            # Run in thread to not block websocket
            t = threading.Thread(target=_send_telegram, args=(token, chat_id, msg))
            t.daemon = True  # FIX: Make daemon so it doesn't block Ctrl+C exit
            t.start()
        except Exception:
            pass

def _send_telegram(token, chat_id, msg):
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        requests.post(url, data={"chat_id": chat_id, "text": msg}, timeout=5)
    except Exception:
        pass

def round_tick(price):
    return round(price * 20) / 20 # Round to nearest 0.05

# ==========================================
# 🚀 KOTAK STRATEGY CLASS
# ==========================================
class KotakORBStrategy:
    def __init__(self):
        load_dotenv()
        self.client = None
        self.tokens_map = {} # Symbol -> Token
        self.reverse_map = {} # Token -> Symbol

        # Data Management
        self.candles = {} # Symbol -> DataFrame (OHLCV)
        self.live_ticks = {} # Symbol -> List of tick data for current candle
        self.last_candle_time = {} # Symbol -> Timestamp of last closed candle

        # Signal Cache (To avoid recalculating on every tick)
        self.latest_signals = {} # Symbol -> {'adx': val, 'trend': val, 'rsi': val}

        self.orders_placed = {} # Symbol -> Boolean (True if trade taken today)
        self.active_orders = {} # Symbol -> {'SL': order_id, 'TP': order_id}
        self.orb_levels = {} # Symbol -> {'high': x, 'low': y}
        self.token_list_for_sub = []

        # Status Tracking
        self.last_tick_arrival = None
        self.tick_count = 0

        # OCO Management
        self.oco_threads = []
        self.is_shutting_down = False # Flag to manage clean exit
        self.heartbeat_thread = None

    def login(self):
        log("🔑 Logging in to Kotak Neo (using totp_login)...")
        # Matches logic from your working login.py
        consumer_key = os.getenv("NEO_APP_TOKEN") or os.getenv("CONSUMER_KEY")
        mobile = os.getenv("KOTAK_MOBILE_NUMBER")
        ucc = os.getenv("KOTAK_UCC")
        mpin = os.getenv("KOTAK_MPIN")
        totp_secret = os.getenv("TOTP_SECRET")

        if not all([consumer_key, mobile, ucc, mpin, totp_secret]):
            log("❌ Missing credentials in .env")
            sys.exit(1)

        try:
            # Initialize Client
            self.client = NeoAPI(environment='prod', consumer_key=consumer_key)

            # Generate TOTP
            totp = pyotp.TOTP(totp_secret).now()

            # Step 1: TOTP Login (Matching login.py)
            resp = self.client.totp_login(mobile_number=mobile, ucc=ucc, totp=totp)
            if isinstance(resp, dict) and 'error' in resp:
                 raise Exception(f"TOTP Login failed: {resp}")

            # Step 2: MPIN Validation (Matching login.py)
            resp = self.client.totp_validate(mpin=mpin)
            if isinstance(resp, dict) and 'error' in resp:
                 raise Exception(f"MPIN Validation failed: {resp}")

            log("✅ Login Successful.")

        except Exception as e:
            log(f"❌ Login Failed: {e}")
            sys.exit(1)

    def load_scrip_master(self):
        log("📥 Loading Scrip Master (this may take a moment)...")
        try:
            scrip_data = self.client.scrip_master(exchange_segment="nse_cm")

            # 1. Handle URL response (which is the standard for Kotak Neo v2)
            if isinstance(scrip_data, str):
                log("   > Downloading Scrip Master CSV...")
                import io
                resp = requests.get(scrip_data)
                resp.raise_for_status()
                # Use pandas to parse CSV efficiently
                df_scrips = pd.read_csv(io.StringIO(resp.text), low_memory=False)
                # Clean header names (strip whitespace) just in case
                df_scrips.columns = df_scrips.columns.str.strip()
                # Convert to dict for compatibility with existing loop
                scrips = df_scrips.to_dict('records')

            # 2. Handle List response (if legacy or direct)
            elif isinstance(scrip_data, list):
                scrips = scrip_data
            else:
                 raise ValueError(f"Unknown Scrip Master response type: {type(scrip_data)}")

            for scrip in scrips:
                symbol_name = scrip.get('pSymbolName') or scrip.get('pSymbol')
                trading_symbol = scrip.get('pTrdSymbol')
                token = scrip.get('pScripRefKey')

                if not token or not trading_symbol: continue

                clean_sym_from_list = lambda s: s.split(':')[1]

                for s in SYMBOLS_TO_TRADE:
                    target_sym = clean_sym_from_list(s)
                    if target_sym == trading_symbol or target_sym == symbol_name:
                         self.tokens_map[s] = str(token)
                         self.reverse_map[str(token)] = s
                         self.token_list_for_sub.append({"instrument_token": str(token), "exchange_segment": "nse_cm"})
                         self.live_ticks[s] = []
                         # Init empty signal for symbol
                         self.latest_signals[s] = None

            log(f"✅ Loaded {len(self.token_list_for_sub)} tokens for monitoring.")

        except Exception as e:
            log(f"❌ Failed to load Scrip Master: {e}")
            sys.exit(1)

    def fetch_warmup_data(self):
        log("🔥 Fetching warm-up data from Local Storage (Zerodha CSVs)...")

        for symbol, token in self.tokens_map.items():
            # Format: NSE:RELIANCE -> NSE_RELIANCE.csv
            filename = f"{symbol.replace(':', '_')}.csv"
            filepath = os.path.join(DATA_DIR, filename)

            if not os.path.exists(filepath):
                log(f"⚠️ Warning: No CSV found for {symbol}. Starting fresh.")
                self.candles[symbol] = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
                continue

            try:
                # Load CSV
                df = pd.read_csv(filepath)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)

                # Standardize columns
                df.rename(columns={'date': 'time'}, inplace=True)

                # Store
                self.candles[symbol] = df
                self.last_candle_time[symbol] = df.index[-1]

                # Calculate initial indicators
                self.update_indicators(symbol)

                log(f"✅ Loaded {len(df)} candles for {symbol}")

                # Calculate ORB levels if late start
                orb_start_dt = datetime.combine(datetime.now().date(), ORB_START_TIME)
                orb_end_dt = datetime.combine(datetime.now().date(), ORB_END_TIME)

                if datetime.now() > orb_end_dt:
                    orb_df = df[(df.index >= orb_start_dt) & (df.index < orb_end_dt)]
                    if not orb_df.empty:
                        orb_high = orb_df['high'].max()
                        orb_low = orb_df['low'].min()
                        self.orb_levels[symbol] = {'high': orb_high, 'low': orb_low}
                        log(f"   📊 Past ORB {symbol}: High={orb_high}, Low={orb_low}")

            except Exception as e:
                log(f"❌ Error loading CSV for {symbol}: {e}")
                self.candles[symbol] = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

    def start_websocket(self):
        log("wss Connecting to WebSocket...")

        def on_message(message):
            # --- CRITICAL FIX: Unwrap the Kotak Data Packet ---
            # Kotak sends data as: {'type': 'stock_feed', 'data': [...]}

            if isinstance(message, dict):
                # Check for Stock Feed
                if message.get('type') == 'stock_feed':
                    real_data = message.get('data')
                    if isinstance(real_data, list):
                        for tick in real_data:
                            self.process_tick(tick)
                    elif isinstance(real_data, dict):
                        self.process_tick(real_data)

                # Check for Order Feed (To update order status)
                elif message.get('type') == 'order_feed':
                    # log(f"📦 Order Update: {message.get('data')}")
                    pass # Can implement order status logic here

                # Fallback: Maybe it IS a direct tick (rare but possible in some modes)
                elif 'tk' in message and 'ltp' in message:
                    self.process_tick(message)

            elif isinstance(message, list):
                for item in message:
                    if isinstance(item, dict) and 'tk' in item:
                        self.process_tick(item)

        def on_error(error):
            # Only print real errors, ignore trivial close notices
            if "socket is already closed" not in str(error):
                log(f"wss Error: {error}")

        def on_open(msg):
            log("wss Connected. Subscribing...")
            # Subscriptions are handled by the library's on_open or explicitly below

        # FIX: Auto-Reconnect Logic in on_close
        def on_close(msg):
            if not self.is_shutting_down:
                log("⚠️ wss Connection Closed Unexpectedly. Reconnecting in 5s...")
                time.sleep(5)
                try:
                    self.client.subscribe(instrument_tokens=self.token_list_for_sub)
                    # Also re-subscribe to order feed
                    self.client.subscribe_to_orderfeed()
                except Exception as e:
                    log(f"❌ Reconnect attempt failed: {e}")
            else:
                log("wss Connection Closed.")

        self.client.on_message = on_message
        self.client.on_error = on_error
        self.client.on_open = on_open
        self.client.on_close = on_close

        # This call initiates the connection AND subscribes
        log(f"📋 Subscribing to {len(self.token_list_for_sub)} tokens.")
        self.client.subscribe(instrument_tokens=self.token_list_for_sub)

        # Also subscribe to Order Feed as per best practices
        try:
            self.client.subscribe_to_orderfeed()
            log("wss Subscribed to Order Feed.")
        except Exception as e:
            log(f"⚠️ Failed to subscribe to order feed: {e}")

    def process_tick(self, tick):
        # Safety check for malformed ticks
        if not isinstance(tick, dict): return

        token = tick.get('tk')
        ltp = tick.get('ltp')

        if not token or not ltp: return

        # Update last tick time for heartbeat
        self.last_tick_arrival = datetime.now()
        self.tick_count += 1

        token = str(token)
        symbol = self.reverse_map.get(token)
        if not symbol: return

        ltp = float(ltp)
        timestamp = datetime.now() # Use local time for aggregation

        # 1. Append tick
        self.live_ticks[symbol].append({'time': timestamp, 'price': ltp})

        # 2. Check Candle Aggregation (5-min)
        # This will now trigger indicator updates ONLY when a candle closes
        self.aggregate_candles(symbol)

        # 3. Check Strategy Trigger
        # This is now LIGHTWEIGHT (no calculations, just checks)
        self.check_orb_strategy(symbol, ltp)

    def aggregate_candles(self, symbol):
        ticks = self.live_ticks[symbol]
        if not ticks: return

        last_tick_time = ticks[-1]['time']

        minute_floored = (last_tick_time.minute // 5) * 5
        current_candle_open_time = last_tick_time.replace(minute=minute_floored, second=0, microsecond=0)

        last_stored = self.last_candle_time.get(symbol)

        if last_stored and current_candle_open_time > last_stored:
            # The previous candle is complete.
            prev_open_time = current_candle_open_time - timedelta(minutes=5)
            candle_ticks = [t['price'] for t in ticks if t['time'] >= prev_open_time and t['time'] < current_candle_open_time]

            if candle_ticks:
                new_candle = {
                    'open': candle_ticks[0],
                    'high': max(candle_ticks),
                    'low': min(candle_ticks),
                    'close': candle_ticks[-1],
                    'volume': 0
                }

                df = self.candles[symbol]
                new_row = pd.DataFrame([new_candle], index=[prev_open_time])
                self.candles[symbol] = pd.concat([df, new_row])
                self.last_candle_time[symbol] = current_candle_open_time
                self.live_ticks[symbol] = [t for t in ticks if t['time'] >= current_candle_open_time]
                log(f"🕯️ New Candle {symbol} @ {prev_open_time.time()}: C={new_candle['close']}")

                # PERFORMANCE FIX: Calculate indicators ONLY here (once per 5 mins)
                self.update_indicators(symbol)

    def update_indicators(self, symbol):
        """Calculate technical indicators on the closed candles and cache them."""
        df = self.candles.get(symbol)
        if df is None or df.empty: return

        try:
            df_calc = df.copy()
            df_calc['adx'] = calculate_adx(df_calc, ADX_PERIOD)
            df_calc = calculate_supertrend(df_calc, SUPERTREND_PERIOD, SUPERTREND_MULTIPLIER)
            df_calc['rsi'] = calculate_rsi(df_calc['close'], RSI_PERIOD)

            last_row = df_calc.iloc[-1]

            self.latest_signals[symbol] = {
                'adx': last_row['adx'],
                'trend': last_row['trend'],
                'rsi': last_row['rsi']
            }
            # log(f"   📊 Indicators Updated {symbol}: RSI={last_row['rsi']:.2f}")
        except Exception as e:
            # log(f"⚠️ Indicator calculation error for {symbol}: {e}")
            pass

    def check_orb_strategy(self, symbol, ltp):
        if self.orders_placed.get(symbol, False): return

        now = datetime.now()
        current_time = now.time()

        # ORB Range Definition
        if current_time < ORB_END_TIME:
            if symbol not in self.orb_levels:
                self.orb_levels[symbol] = {'high': 0, 'low': 999999}

            if ltp > self.orb_levels[symbol]['high']:
                self.orb_levels[symbol]['high'] = ltp
            if ltp < self.orb_levels[symbol]['low']:
                self.orb_levels[symbol]['low'] = ltp
            return

        # Breakout Check
        if current_time >= ORB_END_TIME and current_time < dtime(15, 15):
            levels = self.orb_levels.get(symbol)
            if not levels: return

            orb_high = levels['high']
            orb_low = levels['low']

            # --- OPTIMIZED CHECK ---
            # Instead of calculating, we just read the cached values from update_indicators
            indicators = self.latest_signals.get(symbol)
            if not indicators: return # Not enough data yet

            adx = indicators['adx']
            supertrend_dir = indicators['trend']
            rsi = indicators['rsi']

            signal = None
            if ltp > orb_high:
                if adx > ADX_THRESHOLD and supertrend_dir == 1 and rsi > RSI_UPPER:
                    signal = "BUY"
            elif ltp < orb_low:
                if adx > ADX_THRESHOLD and supertrend_dir == -1 and rsi < RSI_LOWER:
                    signal = "SELL"

            if signal:
                self.execute_trade(symbol, signal, ltp, orb_high, orb_low)

    def execute_trade(self, symbol, signal, price, orb_high, orb_low):
        log(f"⚡ SIGNAL DETECTED: {signal} {symbol} @ {price}")

        entry_price = price
        if signal == "BUY":
            sl_dist = entry_price * (SL_PERCENT / 100)
            sl_price = entry_price - sl_dist
            tp_dist = entry_price * (TARGET_PERCENT / 100)
            tp_price = entry_price + tp_dist
        else:
            sl_dist = entry_price * (SL_PERCENT / 100)
            sl_price = entry_price + sl_dist
            tp_dist = entry_price * (TARGET_PERCENT / 100)
            tp_price = entry_price - tp_dist

        risk_per_share = abs(entry_price - sl_price)
        if risk_per_share == 0: return

        qty = int(RISK_PER_TRADE / risk_per_share)
        if qty < 1: qty = 1

        log(f"📝 Placing Order: {signal} {qty} Qty | SL: {sl_price:.2f} | TP: {tp_price:.2f}")

        try:
            # 1. Entry Order
            t_type = "B" if signal == "BUY" else "S"
            # Ensure AMO is set to "NO" explicitly as per documentation
            entry_resp = self.client.place_order(
                exchange_segment="nse_cm", product="MIS", price="0", order_type="MKT",
                quantity=str(qty), validity="DAY", trading_symbol=symbol.split(':')[1],
                transaction_type=t_type, amo="NO"
            )

            # --- VITAL CHECK: Did the order actually go through? ---
            if isinstance(entry_resp, dict) and entry_resp.get('stat') in ["Ok", "ok", "OK"]:
                log(f"✅ Entry Order Success. ID: {entry_resp.get('nOrdNo')}")
            else:
                log(f"❌ Entry Order FAILED. Reason: {entry_resp}")
                telegram_alert(f"❌ Entry Failed {symbol}: {entry_resp}")
                return # STOP here if entry failed!

            # 2. SL Order
            sl_t_type = "S" if signal == "BUY" else "B"
            # FIX: Trigger Price must be a string and price must be "0" for SL-M
            sl_price_str = "0"
            sl_order_type = "SL-M"

            sl_resp = self.client.place_order(
                exchange_segment="nse_cm", product="MIS",
                price=sl_price_str,
                order_type=sl_order_type,
                quantity=str(qty), validity="DAY", trading_symbol=symbol.split(':')[1],
                transaction_type=sl_t_type,
                trigger_price=str(round_tick(sl_price)),
                amo="NO"
            )

            # 3. TP Order
            tp_resp = self.client.place_order(
                exchange_segment="nse_cm", product="MIS",
                price=str(round_tick(tp_price)), order_type="L",
                quantity=str(qty), validity="DAY", trading_symbol=symbol.split(':')[1],
                transaction_type=sl_t_type,
                amo="NO"
            )

            # Capture IDs for OCO monitoring
            sl_id = None
            tp_id = None

            # Handling API response variations (can be dict with 'nOrdNo' or 'ordId')
            if isinstance(sl_resp, dict):
                 sl_id = sl_resp.get('nOrdNo') or sl_resp.get('ordId')
            if isinstance(tp_resp, dict):
                 tp_id = tp_resp.get('nOrdNo') or tp_resp.get('ordId')

            self.orders_placed[symbol] = True

            msg = f"🚀 Trade Executed: {symbol} {signal} @ {price}\nQty: {qty}\nSL: {sl_price:.2f}\nTP: {tp_price:.2f}"
            telegram_alert(msg)

            if sl_id and tp_id:
                log(f"🛡️ Starting OCO Monitor for {symbol} (SL: {sl_id}, TP: {tp_id})")
                t = threading.Thread(target=self.monitor_oco, args=(symbol, str(sl_id), str(tp_id)))
                t.daemon = True
                t.start()

        except Exception as e:
            log(f"❌ Order Placement Failed: {e}")
            telegram_alert(f"⚠️ Order Failed: {symbol} - {e}")

    def monitor_oco(self, symbol, sl_id, tp_id):
        # Poll order history to check if SL or TP is executed
        # If one executes, cancel the other.
        # This is a basic fail-safe implementation.
        log(f"Monitoring OCO for {symbol} started.")
        while True:
            time.sleep(5) # Poll every 5 seconds
            try:
                # We need to fetch order report or history
                # client.order_report() returns list of all orders

                report = self.client.order_report()
                if not report or not isinstance(report, dict): continue

                data = report.get('data', [])
                if not data: continue

                # Check status
                sl_status = None
                tp_status = None

                for order in data:
                    o_id = order.get('nOrdNo')
                    status = order.get('ordSt')

                    if str(o_id) == sl_id:
                        sl_status = status
                    elif str(o_id) == tp_id:
                        tp_status = status

                # Check for completion (trd = Traded/Complete)
                # Kotak status: 'trd', 'can', 'rej'

                sl_done = sl_status in ['trd', 'complete', 'filled']
                tp_done = tp_status in ['trd', 'complete', 'filled']

                if sl_done and not tp_done:
                    log(f"🛑 SL Hit for {symbol}. Cancelling TP ({tp_id})...")
                    self.client.cancel_order(order_id=tp_id)
                    break

                if tp_done and not sl_done:
                    log(f"🎯 TP Hit for {symbol}. Cancelling SL ({sl_id})...")
                    self.client.cancel_order(order_id=sl_id)
                    break

                # Also break if both cancelled or rejected
                sl_dead = sl_status in ['can', 'rej', 'cancelled', 'rejected']
                tp_dead = tp_status in ['can', 'rej', 'cancelled', 'rejected']

                if sl_dead and tp_dead:
                    log(f"⚠️ Both orders closed for {symbol}. OCO monitor ending.")
                    break

            except Exception as e:
                log(f"⚠️ OCO Monitor Error: {e}")
                time.sleep(10) # Backoff

    def run(self):
        self.login()
        self.load_scrip_master()
        self.fetch_warmup_data()
        self.start_websocket()

        log("🟢 Strategy Running. Waiting for ticks...")
        # IMPLEMENTING HEARTBEAT LOG LOOP
        while True:
            time.sleep(60) # Log every 1 minute
            last_tick_time = self.last_tick_arrival.strftime('%H:%M:%S') if self.last_tick_arrival else "No Ticks Yet"
            log(f"💓 Heartbeat: {self.tick_count} total ticks received. Last tick at: {last_tick_time}")

    def shutdown(self):
        # Method to cleanly signal shutdown
        self.is_shutting_down = True

if __name__ == "__main__":
    strategy = KotakORBStrategy()
    try:
        strategy.run()
    except KeyboardInterrupt:
        strategy.shutdown() # Signal shutdown before exiting
        print("\n🔴 Strategy stopped by user.")
        sys.exit(0)