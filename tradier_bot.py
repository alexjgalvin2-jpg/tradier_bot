"""
Tradier Options Bot — Options momentum trading via Tradier
Paper mode: Tradier sandbox (no real money)
Live mode:  Real Tradier brokerage account

Strategy:
  • Bullish signal  → buy call (25–50 DTE, delta ~0.40)
  • Bearish signal  → buy put  (25–50 DTE, delta ~0.40)
  • Exit at +50% premium gain or -30% loss
"""

import os
import time
import json
import logging
import requests
from datetime import datetime, date, timedelta
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
from whale_tracker import (
    get_all_trades, get_whale_rankings, get_whale_signals,
    get_leaderboard_text, get_recent_whale_activity_text,
    WHALE_TOP_N
)
from risk_filters import (
    passes_all_filters, vix_premium_multiplier,
    vix_status_text, sector_exposure_text, get_vix
)

# =============================================================================
#  CONFIG
# =============================================================================
TRADIER_TOKEN   = os.getenv("TRADIER_TOKEN", "")
TRADIER_ACCOUNT = os.getenv("TRADIER_ACCOUNT", "")
TELEGRAM_TOKEN  = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

PAPER_MODE = False  # 💰 LIVE MODE — real money

# Tradier sandbox = paper, api = live
BASE_URL = "https://sandbox.tradier.com/v1" if PAPER_MODE else "https://api.tradier.com/v1"

MAX_PREMIUM      = 100.0   # $100 per trade — conservative for real money start
MAX_POSITIONS    = 10      # max 10 open at once — controlled exposure
TAKE_PROFIT_PCT  = 25.0    # exit at +25%
STOP_LOSS_PCT    = 20.0    # exit at -20%
SCAN_INTERVAL    = 300     # 5 minutes between full scans
DAILY_SUMMARY_HOUR = 16    # send daily summary at 4 PM

DAILY_LOSS_LIMIT      = 300.0   # halt trading if down $300 in a day — protects real money
ACCOUNT_MINIMUM       = 2500.0  # halt ALL trading if live cash balance drops below this (50% of $5k deposit)
SYMBOL_COOLDOWN_HOURS = 24

TARGET_DTE_MIN   = 25
TARGET_DTE_MAX   = 50
TARGET_DELTA_MIN = 0.30
TARGET_DELTA_MAX = 0.55

RSI_PERIOD  = 14
RSI_BULL    = 55
RSI_BEAR    = 45
SMA_PERIOD  = 20
VOLUME_MULT = 1.3

SIGNAL_MODE      = "all"
STRADDLE_MODE    = True
PARALLEL_SCAN    = True

# =============================================================================
#  TRAILING STOP SETTINGS
# =============================================================================
TRAIL_ACTIVATE_PCT = 10.0  # start trailing after position is up 10%
TRAIL_DISTANCE_PCT = 8.0   # trail stop sits 8% below the highest price reached
# Example: buy at $1.00 → rises to $1.50 (+50%) → trail stop at $1.50 * 0.92 = $1.38
#          if price drops to $1.38 → EXIT locking in +38% instead of waiting for +25% fixed TP

# =============================================================================
#  LADDERING SETTINGS
# =============================================================================
LADDER_MODE      = True    # scale into positions in tranches
LADDER_TRANCHES  = 3       # number of entries per signal
LADDER_SIZES     = [0.5, 0.3, 0.2]   # 50% / 30% / 20% of MAX_PREMIUM per tranche
LADDER_TRIGGER   = 5.0     # enter next tranche when position up this % (confirms trend)

# =============================================================================
#  POLITICIAN TRADE TRACKING
# =============================================================================
POLITICIAN_MODE        = True   # use Congressional trade disclosures as signal boost
POLITICIAN_BOOST_RSI   = 3      # reduce RSI requirement by this many points if politicians buying
POLITICIAN_LOOKBACK_DAYS = 30   # only count politician trades from last 30 days
POLITICIAN_MIN_TRADES  = 2      # minimum politician buys to count as a signal

# =============================================================================
#  WHEEL STRATEGY SETTINGS
# =============================================================================
WHEEL_MODE       = True    # enable wheel strategy alongside momentum trades
WHEEL_SYMBOLS    = [       # stable stocks good for the wheel
    "AAPL", "MSFT", "SPY", "QQQ", "NVDA", "AMZN", "GOOGL"
]
WHEEL_DELTA      = 0.30    # sell puts/calls at ~0.30 delta (30% chance of assignment)
WHEEL_DTE_MIN    = 20      # sell options with 20-45 DTE
WHEEL_DTE_MAX    = 45
WHEEL_PREMIUM    = 300.0   # max premium to sell per wheel position

# =============================================================================
#  SPACEX IPO STRADDLE SETTINGS
# =============================================================================
SPACEX_MODE      = True    # enable SpaceX IPO straddle strategy
SPACEX_SYMBOLS   = [       # stocks that move with SpaceX IPO
    "RKLB", "ASTS", "BA", "LMT", "ARKK"
]
SPACEX_STRADDLE_DTE = 7    # buy straddles with 7 DTE around IPO date
SPACEX_IPO_DATE  = "2026-12-01"  # update when confirmed — placeholder for now

# =============================================================================
#  ADAPTIVE LEARNING SETTINGS
# =============================================================================
PERF_LOOKBACK            = 10   # how many recent trades per symbol to analyse
MIN_SYMBOL_WIN_RATE      = 0.35 # skip symbol if its win rate is below 35%
MIN_SYMBOL_TRADES        = 3    # minimum trades before win rate is applied
ADAPTIVE_RSI_THRESHOLD   = 0.40 # if overall win rate below 40%, tighten RSI by 3pts
MAX_CONSEC_LOSSES_SYMBOL = 3    # blacklist symbol after 3 losses in a row

SYMBOLS = [
    # Mega cap tech
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
    # Semiconductors
    "AMD",  "ARM",  "SMCI", "INTC", "QCOM", "MU",
    # High momentum
    "COIN", "PLTR", "CRWD", "SHOP", "MSTR", "SOFI", "SQ",
    # Growth / volatile
    "NFLX", "UBER", "SNAP", "HOOD", "RBLX", "DKNG", "RIVN",
    # ETFs (broader market)
    "SPY",  "QQQ",  "IWM",  "ARKK", "SOXL",
    # Finance / banks
    "JPM",  "GS",   "BAC",  "MS",
    # Energy / commodities
    "XOM",  "CVX",  "OXY",
    # Healthcare / biotech
    "LLY",  "MRNA", "BNTX",
    # Consumer
    "AMZN", "WMT",  "TGT",  "NKE",
]
# Remove duplicates while preserving order
SYMBOLS = list(dict.fromkeys(SYMBOLS))

DATA_DIR       = "/app/data"
os.makedirs(DATA_DIR, exist_ok=True)
STATE_FILE     = os.path.join(DATA_DIR, "tradier_state.json")
TRADE_LOG_FILE = os.path.join(DATA_DIR, "tradier_trades.json")

# =============================================================================
#  LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("tradier_bot.log", encoding="utf-8"),
    ]
)
log = logging.getLogger("TradierBot")


# =============================================================================
#  TELEGRAM
# =============================================================================
PLATFORM = "📈 Tradier Options"

def send_telegram(msg: str):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": f"[{PLATFORM}]\n{msg}"}, timeout=10)
    except Exception as e:
        log.warning("Telegram failed: %s", e)


_last_update_id = 0

def check_telegram_commands(bot) -> None:
    """Poll Telegram for /report command and reply with current status."""
    global _last_update_id
    try:
        url  = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
        resp = requests.get(url, params={"offset": _last_update_id + 1, "timeout": 2}, timeout=10)
        data = resp.json()
        for update in data.get("result", []):
            _last_update_id = update["update_id"]
            msg = update.get("message", {})
            text = msg.get("text", "").strip().lower()
            if text == "/report":
                try:
                    positions = bot.state.get("positions", {})
                    trade_log = []
                    try:
                        with open(TRADE_LOG_FILE) as f:
                            trade_log = json.load(f)
                    except Exception:
                        pass
                    opens   = [t for t in trade_log if t.get("action") == "open"  and not t.get("paper", False)]
                    closes  = [t for t in trade_log if t.get("action") == "close" and not t.get("paper", False)]
                    winners = [t for t in closes if t.get("pnl_usd", 0) > 0]
                    losers  = [t for t in closes if t.get("pnl_usd", 0) <= 0]
                    total_pnl = sum(t.get("pnl_usd", 0) for t in closes)
                    win_rate  = (len(winners) / len(closes) * 100) if closes else 0
                    best  = max(closes, key=lambda t: t.get("pnl_usd", 0)) if closes else None
                    worst = min(closes, key=lambda t: t.get("pnl_usd", 0)) if closes else None

                    pos_lines = ""
                    for opt_sym, pos in positions.items():
                        pos_lines += f"  • {pos['symbol']} {pos['option_type']} ${pos['strike']} exp {pos['expiration']}\n"

                    # Adaptive learning stats
                    perf = get_performance_stats()
                    rsi_adj = perf.get("rsi_adjustment", 0)
                    blocked = [s for s in SYMBOLS if not symbol_is_allowed(s, perf)]
                    cooldown_syms = [s for s in bot.symbol_cooldowns.keys()]

                    send_telegram(
                        f"📊 Tradier Options Report\n"
                        f"Mode: {'PAPER 🧪' if PAPER_MODE else 'LIVE 💰'}\n"
                        f"─────────────────\n"
                        f"Open positions: {len(positions)}/{MAX_POSITIONS}\n"
                        f"{pos_lines if pos_lines else '  None\n'}"
                        f"─────────────────\n"
                        f"Total trades opened: {len(opens)}\n"
                        f"Total trades closed: {len(closes)}\n"
                        f"Winners: {len(winners)}  Losers: {len(losers)}\n"
                        f"Win rate: {win_rate:.0f}%\n"
                        f"─────────────────\n"
                        f"Total P&L: ${total_pnl:+.2f}\n"
                        + (f"Best trade: {best['symbol']} ${best['pnl_usd']:+.2f}\n" if best else "")
                        + (f"Worst trade: {worst['symbol']} ${worst['pnl_usd']:+.2f}\n" if worst else "No closed trades yet\n")
                        + f"─────────────────\n"
                        + f"🧠 Adaptive Learning\n"
                        + f"RSI tightened: {'Yes +3pts' if rsi_adj else 'No'}\n"
                        + f"Blocked symbols: {', '.join(blocked) if blocked else 'None'}\n"
                        + f"On cooldown: {', '.join(cooldown_syms) if cooldown_syms else 'None'}\n"
                        + f"─────────────────\n"
                        + f"🏛️ Politician Tracking: {'ON' if POLITICIAN_MODE else 'OFF'}\n"
                        + f"Trailing stop: activates at +{TRAIL_ACTIVATE_PCT}%, trails {TRAIL_DISTANCE_PCT}%\n"
                        + f"Laddering: {'ON' if LADDER_MODE else 'OFF'} ({LADDER_TRANCHES} tranches)\n"
                        + f"─────────────────\n"
                        + f"📊 Risk Status\n"
                        + f"{vix_status_text()}\n"
                        + f"Sector exposure:\n{sector_exposure_text(positions)}"
                    )
                except Exception as e:
                    send_telegram(f"⚠️ Report error: {e}")
                log.info("Sent /report to Telegram")

            elif text == "/accounthistory":
                try:
                    trade_log = []
                    try:
                        with open(TRADE_LOG_FILE) as f:
                            trade_log = json.load(f)
                    except Exception:
                        pass
                    closes = [t for t in trade_log
                              if t.get("action") == "close" and not t.get("paper", False)]
                    if not closes:
                        send_telegram("📜 Tradier Account History\nNo live trades yet.")
                    else:
                        running_pnl = 0.0
                        # Split into chunks to avoid Telegram 4096 char limit
                        chunk = f"📜 Tradier Account History\n{'─'*25}\n"
                        messages = []
                        for i, t in enumerate(closes, 1):
                            pnl = t.get("pnl_usd", 0)
                            running_pnl += pnl
                            ts  = t.get("timestamp", "")[:10]
                            emoji = "✅" if pnl > 0 else "❌"
                            line = (f"{emoji} #{i} {ts} | {t.get('symbol','?')} "
                                    f"{t.get('option_type','?')} | "
                                    f"P&L: ${pnl:+.2f} | Running: ${running_pnl:+.2f}\n")
                            if len(chunk) + len(line) > 3800:
                                messages.append(chunk)
                                chunk = f"📜 History (cont.)\n{'─'*25}\n"
                            chunk += line
                        chunk += f"{'─'*25}\nFinal P&L: ${running_pnl:+.2f}"
                        messages.append(chunk)
                        for m in messages:
                            send_telegram(m)
                except Exception as e:
                    send_telegram(f"⚠️ History error: {e}")
                log.info("Sent /accounthistory to Telegram")

            elif text == "/politicians":
                try:
                    send_telegram("🔄 Building whale rankings... this may take a moment.")
                    leaderboard = get_leaderboard_text(top_n=10)
                    send_telegram(leaderboard)
                    recent = get_recent_whale_activity_text()
                    send_telegram(recent)
                except Exception as e:
                    send_telegram(f"⚠️ Whale lookup error: {e}")
                log.info("Sent /politicians to Telegram")

            elif text == "/whales":
                try:
                    signals = get_whale_signals(lookback_days=7)
                    if not signals:
                        send_telegram("🐋 No whale trades in the last 7 days.")
                    else:
                        lines = f"🐋 Active Whale Signals\n{'─'*25}\n"
                        for s in signals[:10]:
                            emoji = "📈" if s["signal"] == "CALL" else "📉"
                            lines += (f"{emoji} {s['symbol']} {s['signal']}\n"
                                      f"   {s['politician']} (#{s['rank']}) • {s['date']}\n")
                        send_telegram(lines)
                except Exception as e:
                    send_telegram(f"⚠️ Whale signal error: {e}")
                log.info("Sent /whales to Telegram")

    except Exception as e:
        log.debug("check_telegram_commands failed: %s", e)


# =============================================================================
#  TRADIER API
# =============================================================================
class TradierAPI:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {TRADIER_TOKEN}",
            "Accept":        "application/json",
        })

    def _get(self, path: str, params: dict = None) -> Optional[dict]:
        try:
            r = self.session.get(f"{BASE_URL}{path}", params=params, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            log.warning("GET %s failed: %s", path, e)
            return None

    def _post(self, path: str, data: dict = None) -> Optional[dict]:
        try:
            r = self.session.post(f"{BASE_URL}{path}", data=data, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            log.warning("POST %s failed: %s", path, e)
            return None

    def get_quote(self, symbol: str) -> Optional[dict]:
        data = self._get("/markets/quotes", {"symbols": symbol, "greeks": "false"})
        if data:
            q = data.get("quotes", {}).get("quote")
            if isinstance(q, dict):
                return q
        return None

    def get_history(self, symbol: str, days: int = 60) -> Optional[pd.DataFrame]:
        start = (date.today() - timedelta(days=days)).strftime("%Y-%m-%d")
        end   = date.today().strftime("%Y-%m-%d")
        data  = self._get("/markets/history", {
            "symbol": symbol, "interval": "daily",
            "start": start,   "end": end,
        })
        if not data:
            return None
        history = data.get("history", {})
        if not history or history == "null":
            return None
        days_data = history.get("day", [])
        if isinstance(days_data, dict):
            days_data = [days_data]
        if not days_data:
            return None
        df = pd.DataFrame(days_data)
        df["date"]   = pd.to_datetime(df["date"])
        df["close"]  = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)
        return df.sort_values("date").reset_index(drop=True)

    def get_expirations(self, symbol: str) -> list:
        data = self._get("/markets/options/expirations", {
            "symbol": symbol, "includeAllRoots": "true",
        })
        if data:
            exps = data.get("expirations", {})
            if exps and exps != "null":
                dates = exps.get("date", [])
                return dates if isinstance(dates, list) else [dates]
        return []

    def get_chain(self, symbol: str, expiration: str) -> list:
        data = self._get("/markets/options/chains", {
            "symbol": symbol, "expiration": expiration, "greeks": "true",
        })
        if data:
            options = data.get("options", {})
            if options and options != "null":
                chain = options.get("option", [])
                return chain if isinstance(chain, list) else [chain]
        return []

    def get_option_quote(self, option_symbol: str) -> Optional[dict]:
        data = self._get("/markets/quotes", {"symbols": option_symbol})
        if data:
            q = data.get("quotes", {}).get("quote")
            if isinstance(q, dict):
                return q
        return None

    def get_account_balance(self) -> float:
        data = self._get(f"/accounts/{TRADIER_ACCOUNT}/balances")
        if data:
            bal = data.get("balances", {})
            # Prefer total_cash (actual deposited cash) over buying power
            # For cash accounts: bal["cash"]["cash_available"]
            # For margin accounts: bal["margin"]["stock_buying_power"] is 2x cash — avoid
            total_cash = bal.get("total_cash")
            if total_cash is not None:
                return float(total_cash)
            cash_block = bal.get("cash") or {}
            if cash_block.get("cash_available") is not None:
                return float(cash_block["cash_available"])
            # Last resort
            return float(bal.get("total_equity", 0))
        return 0.0

    def place_option_order(self, symbol: str, option_symbol: str,
                           side: str, quantity: int,
                           limit_price: float = None) -> Optional[dict]:
        """side: buy_to_open or sell_to_close. Uses limit orders (required by Tradier for options)."""
        if PAPER_MODE:
            log.info("📄 PAPER ORDER: %s %s x%d", side, option_symbol, quantity)
            return {"id": f"paper-{int(time.time())}", "status": "ok", "paper": True}

        if limit_price is None or limit_price <= 0:
            log.warning("place_option_order: no valid limit_price for %s — skipping", option_symbol)
            return None

        price = round(limit_price, 2)
        data = self._post(f"/accounts/{TRADIER_ACCOUNT}/orders", {
            "class":         "option",
            "symbol":        symbol,
            "option_symbol": option_symbol,
            "side":          side,
            "quantity":      str(quantity),
            "type":          "limit",
            "price":         str(price),
            "duration":      "day",
        })
        log.info("Limit order: %s %s x%d @ $%.2f", side, option_symbol, quantity, price)
        return data


# =============================================================================
#  TECHNICAL SIGNALS
# =============================================================================
def compute_rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff().dropna()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(com=period - 1, min_periods=period).mean().iloc[-1]
    avg_l = loss.ewm(com=period - 1, min_periods=period).mean().iloc[-1]
    if avg_l == 0:
        return 100.0
    return 100 - (100 / (1 + avg_g / avg_l))


def get_signal(api: TradierAPI, symbol: str, rsi_adj: int = 0) -> Optional[str]:
    """
    Returns 'CALL', 'PUT', or None.
    rsi_adj: adaptive tightening — added to RSI_BULL, subtracted from RSI_BEAR
             when overall win rate is low (bot learns to be more selective)
    """
    df = api.get_history(symbol, days=60)
    if df is None or len(df) < SMA_PERIOD + 5:
        log.debug("%s: not enough history", symbol)
        return None

    close     = df["close"]
    volume    = df["volume"]
    sma       = close.rolling(SMA_PERIOD).mean().iloc[-1]
    price     = close.iloc[-1]
    rsi       = compute_rsi(close, RSI_PERIOD)
    vol_ma    = volume.rolling(10).mean().iloc[-2]
    vol_now   = volume.iloc[-1]
    vol_ratio = vol_now / vol_ma if vol_ma > 0 else 0

    # Apply adaptive adjustment — tighter when performance is poor
    rsi_bull_threshold = RSI_BULL + rsi_adj
    rsi_bear_threshold = RSI_BEAR - rsi_adj

    log.info("%s  price=%.2f  SMA=%.2f  RSI=%.1f  vol_ratio=%.2f  RSI_thresholds=[>%d,<%d]",
             symbol, price, sma, rsi, vol_ratio, rsi_bull_threshold, rsi_bear_threshold)

    price_bull = price > sma
    price_bear = price < sma
    rsi_bull   = rsi >= rsi_bull_threshold
    rsi_bear   = rsi <= rsi_bear_threshold
    vol_spike  = vol_ratio >= VOLUME_MULT

    if SIGNAL_MODE == "all":
        if price_bull and rsi_bull and vol_spike:
            return "CALL"
        if price_bear and rsi_bear and vol_spike:
            return "PUT"
    else:  # any2 — 2 of 3 conditions
        bull_score = sum([price_bull, rsi_bull, vol_spike])
        bear_score = sum([price_bear, rsi_bear, vol_spike])
        if bull_score >= 2:
            return "CALL"
        if bear_score >= 2:
            return "PUT"

    return None


# =============================================================================
#  OPTIONS SELECTION
# =============================================================================
def pick_expiration(expirations: list) -> Optional[str]:
    today = date.today()
    best  = None
    for exp_str in expirations:
        try:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            dte      = (exp_date - today).days
            if TARGET_DTE_MIN <= dte <= TARGET_DTE_MAX:
                if best is None:
                    best = exp_str
                else:
                    best_dte = (datetime.strptime(best, "%Y-%m-%d").date() - today).days
                    if dte < best_dte:
                        best = exp_str
        except Exception:
            continue
    return best


def pick_option(chain: list, option_type: str, spot: float) -> Optional[dict]:
    """Pick best call/put near 0.40 delta and within budget."""
    filtered = [o for o in chain
                if o.get("option_type", "").lower() == option_type.lower()]
    if not filtered:
        return None

    # Delta-based selection
    candidates = []
    for o in filtered:
        greeks = o.get("greeks") or {}
        delta  = greeks.get("delta")
        ask    = o.get("ask")
        if delta is None or ask is None:
            continue
        delta = abs(float(delta))
        ask   = float(ask)
        if TARGET_DELTA_MIN <= delta <= TARGET_DELTA_MAX and 0 < ask * 100 <= MAX_PREMIUM:
            candidates.append((abs(delta - 0.40), o))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    # Fallback: nearest ATM within budget
    atm = sorted(filtered, key=lambda o: abs(float(o.get("strike", 0)) - spot))
    for o in atm:
        ask = float(o.get("ask") or 0)
        if 0 < ask * 100 <= MAX_PREMIUM:
            return o
    return None


# =============================================================================
#  STATE
# =============================================================================
# =============================================================================
#  ADAPTIVE LEARNING ENGINE
# =============================================================================
def get_performance_stats() -> dict:
    """
    Reads trade log and returns:
    - per-symbol win rates, consecutive losses, avg P&L
    - overall win rate
    - adaptive RSI adjustment
    """
    try:
        with open(TRADE_LOG_FILE) as f:
            all_trades = json.load(f)
    except Exception:
        return {"overall_win_rate": 1.0, "symbols": {}, "rsi_adjustment": 0}

    closes = [t for t in all_trades if t.get("action") == "close"]
    if not closes:
        return {"overall_win_rate": 1.0, "symbols": {}, "rsi_adjustment": 0}

    # Overall win rate
    winners      = [t for t in closes if t.get("pnl_usd", 0) > 0]
    overall_wr   = len(winners) / len(closes) if closes else 1.0

    # Per-symbol stats using last PERF_LOOKBACK trades
    symbol_stats = {}
    symbols_seen = set(t.get("symbol") for t in closes)
    for sym in symbols_seen:
        sym_trades = [t for t in closes if t.get("symbol") == sym][-PERF_LOOKBACK:]
        sym_wins   = [t for t in sym_trades if t.get("pnl_usd", 0) > 0]
        sym_wr     = len(sym_wins) / len(sym_trades) if sym_trades else 1.0
        avg_pnl    = sum(t.get("pnl_usd", 0) for t in sym_trades) / len(sym_trades)

        # Count consecutive losses from the end
        consec_losses = 0
        for t in reversed(sym_trades):
            if t.get("pnl_usd", 0) < 0:
                consec_losses += 1
            else:
                break

        symbol_stats[sym] = {
            "trades":        len(sym_trades),
            "win_rate":      round(sym_wr, 3),
            "avg_pnl":       round(avg_pnl, 2),
            "consec_losses": consec_losses,
        }

    # RSI tightening: if overall win rate is below threshold, add 3pts to RSI requirement
    rsi_adjustment = 3 if overall_wr < ADAPTIVE_RSI_THRESHOLD else 0

    return {
        "overall_win_rate": round(overall_wr, 3),
        "symbols":          symbol_stats,
        "rsi_adjustment":   rsi_adjustment,
    }


def symbol_is_allowed(symbol: str, perf: dict) -> bool:
    """
    Returns False if symbol should be skipped based on recent performance.
    - Too many consecutive losses → blocked
    - Win rate too low (with enough data) → blocked
    """
    stats = perf.get("symbols", {}).get(symbol)
    if not stats:
        return True  # no data yet, allow it

    if stats["consec_losses"] >= MAX_CONSEC_LOSSES_SYMBOL:
        log.info("%s blocked — %d consecutive losses", symbol, stats["consec_losses"])
        return False

    if stats["trades"] >= MIN_SYMBOL_TRADES and stats["win_rate"] < MIN_SYMBOL_WIN_RATE:
        log.info("%s blocked — win rate %.0f%% below %.0f%% minimum",
                 symbol, stats["win_rate"] * 100, MIN_SYMBOL_WIN_RATE * 100)
        return False

    return True


# =============================================================================
#  WHEEL STRATEGY
# =============================================================================
def pick_wheel_option(api: TradierAPI, symbol: str,
                      option_type: str) -> Optional[dict]:
    """
    Pick an option to SELL for the wheel strategy.
    option_type: 'put' (step 1 - cash secured put) or 'call' (step 3 - covered call)
    Targets ~0.30 delta, 20-45 DTE.
    """
    expirations = api.get_expirations(symbol)
    today       = date.today()
    valid_exps  = []
    for exp_str in expirations:
        try:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            dte      = (exp_date - today).days
            if WHEEL_DTE_MIN <= dte <= WHEEL_DTE_MAX:
                valid_exps.append((dte, exp_str))
        except Exception:
            continue
    if not valid_exps:
        return None
    valid_exps.sort(key=lambda x: x[0])
    _, expiration = valid_exps[0]

    chain = api.get_chain(symbol, expiration)
    opts  = [o for o in chain if o.get("option_type", "").lower() == option_type]

    # Find option near 0.30 delta — for puts delta is negative so use abs
    candidates = []
    for o in opts:
        greeks = o.get("greeks") or {}
        delta  = greeks.get("delta")
        bid    = o.get("bid")
        if delta is None or bid is None:
            continue
        delta = abs(float(delta))
        bid   = float(bid)
        if 0.20 <= delta <= 0.40 and bid * 100 <= WHEEL_PREMIUM and bid > 0:
            candidates.append((abs(delta - WHEEL_DELTA), o))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def run_wheel_scan(api: TradierAPI, state: dict) -> list:
    """
    Scan wheel symbols and return list of (symbol, action, option) tuples.
    action = 'sell_put' or 'sell_call'
    """
    opportunities = []
    wheel_positions = state.get("wheel_positions", {})

    for symbol in WHEEL_SYMBOLS:
        # Already have a wheel position on this symbol
        if symbol in wheel_positions:
            pos = wheel_positions[symbol]
            # If assigned (own stock) → sell covered call
            if pos.get("stage") == "assigned":
                opt = pick_wheel_option(api, symbol, "call")
                if opt:
                    opportunities.append((symbol, "sell_call", opt))
            continue

        # No position → sell cash secured put
        opt = pick_wheel_option(api, symbol, "put")
        if opt:
            opportunities.append((symbol, "sell_put", opt))

    return opportunities


# =============================================================================
#  SPACEX IPO STRADDLE
# =============================================================================
def check_spacex_straddle(api: TradierAPI, state: dict) -> list:
    """
    Around SpaceX IPO date, buy straddles (call + put) on proxy symbols.
    Returns list of (symbol, signal) tuples to enter.
    """
    try:
        ipo_date  = datetime.strptime(SPACEX_IPO_DATE, "%Y-%m-%d").date()
        today     = date.today()
        days_away = (ipo_date - today).days

        # Only activate within 14 days of IPO
        if not (0 <= days_away <= 14):
            return []

        log.info("🚀 SpaceX IPO in %d days — scanning straddle opportunities", days_away)

        opportunities = []
        spacex_positions = state.get("spacex_positions", {})

        for symbol in SPACEX_SYMBOLS:
            if symbol in spacex_positions:
                continue  # already in a straddle

            # Buy both call and put (straddle)
            opportunities.append((symbol, "CALL"))
            opportunities.append((symbol, "PUT"))

        if opportunities and days_away <= 14:
            send_telegram(
                f"🚀 SpaceX IPO Alert!\n"
                f"IPO in {days_away} days ({SPACEX_IPO_DATE})\n"
                f"Entering straddles on: {', '.join(SPACEX_SYMBOLS)}\n"
                f"Strategy: profit whether SpaceX goes UP or DOWN 📈📉"
            )

        return opportunities
    except Exception as e:
        log.warning("check_spacex_straddle failed: %s", e)
        return []


# =============================================================================
#  POLITICIAN WHALE INTEGRATION
# =============================================================================
def get_politician_signal(symbol: str, whale_signals: list) -> int:
    """
    Returns RSI adjustment based on whale activity on this symbol.
    If a top whale bought → lower RSI threshold (easier to enter)
    If a top whale sold  → raise RSI threshold (harder to enter)
    """
    for sig in whale_signals:
        if sig["symbol"] == symbol:
            if sig["signal"] == "CALL":
                log.info("🐋 %s: whale %s (rank #%d) bought — boosting signal",
                         symbol, sig["politician"], sig["rank"])
                return -POLITICIAN_BOOST_RSI
            elif sig["signal"] == "PUT":
                log.info("🐋 %s: whale %s (rank #%d) sold — dampening signal",
                         symbol, sig["politician"], sig["rank"])
                return POLITICIAN_BOOST_RSI
    return 0


def load_state() -> dict:
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"positions": {}}


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def log_trade(entry: dict):
    try:
        try:
            with open(TRADE_LOG_FILE) as f:
                trades = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            trades = []
        trades.append(entry)
        with open(TRADE_LOG_FILE, "w") as f:
            json.dump(trades, f, indent=2)
    except Exception as e:
        log.warning("log_trade failed: %s", e)


# =============================================================================
#  MAIN BOT
# =============================================================================
class TradierOptionsBot:
    def __init__(self):
        self.api              = TradierAPI()
        self.state            = load_state()
        self.scan_count       = 0
        self.symbol_cooldowns = {}   # symbol → datetime of last loss, blocks re-entry
        self.trading_halted   = False  # set True when daily loss limit hit

        # Load all-time P&L from trade log — only count live trades (not paper history)
        try:
            with open(TRADE_LOG_FILE) as f:
                _trades = json.load(f)
            self.session_pnl = sum(
                t.get("pnl_usd", 0) for t in _trades
                if t.get("action") == "close" and not t.get("paper", False)
            )
        except Exception:
            self.session_pnl = 0.0

        self.daily_trades     = []
        self.last_summary_day = None
        log.info("TradierOptionsBot initialised — %s mode",
                 "PAPER" if PAPER_MODE else "LIVE")

    def _is_market_open(self, now: datetime = None) -> bool:
        """Returns True if US options market is open (Mon-Fri 9:30-16:00 ET)."""
        if now is None:
            now = datetime.now()
        if now.weekday() >= 5:  # Saturday=5, Sunday=6
            return False
        market_open  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0,  second=0, microsecond=0)
        return market_open <= now <= market_close

    def _today_pnl(self) -> float:
        """Calculate today's P&L from live trades only."""
        try:
            with open(TRADE_LOG_FILE) as f:
                all_trades = json.load(f)
            today = date.today().isoformat()
            return sum(t.get("pnl_usd", 0) for t in all_trades
                       if t.get("action") == "close"
                       and t.get("timestamp", "").startswith(today)
                       and not t.get("paper", False))
        except Exception:
            return 0.0

    def _check_daily_loss_limit(self) -> bool:
        """Returns True if we should halt trading for the day."""
        today_pnl = self._today_pnl()
        if today_pnl <= -DAILY_LOSS_LIMIT:
            if not self.trading_halted:
                self.trading_halted = True
                send_telegram(
                    f"🚨 DAILY LOSS LIMIT HIT\n"
                    f"Today's P&L: ${today_pnl:+.2f}\n"
                    f"Limit: -${DAILY_LOSS_LIMIT:.0f}\n"
                    f"Trading HALTED for today. Resumes tomorrow."
                )
                log.warning("Daily loss limit hit — halting trading for today")
            return True
        return False

    def _check_account_minimum(self) -> bool:
        """Returns True if account balance fell below ACCOUNT_MINIMUM — halt all trading."""
        balance = self.api.get_account_balance()
        if 0 < balance < ACCOUNT_MINIMUM:
            if not self.trading_halted:
                self.trading_halted = True
                send_telegram(
                    f"🚨 ACCOUNT MINIMUM REACHED\n"
                    f"Balance: ${balance:,.2f}\n"
                    f"Minimum threshold: ${ACCOUNT_MINIMUM:,.2f}\n"
                    f"Trading HALTED to protect remaining capital.\n"
                    f"Review your account and restart the bot manually to resume."
                )
                log.warning("Account balance $%.2f below minimum $%.2f — halting all trading",
                            balance, ACCOUNT_MINIMUM)
            return True
        return False

    def _symbol_on_cooldown(self, symbol: str) -> bool:
        """Returns True if symbol lost recently and is on cooldown."""
        if symbol in self.symbol_cooldowns:
            elapsed = (datetime.now() - self.symbol_cooldowns[symbol]).total_seconds() / 3600
            if elapsed < SYMBOL_COOLDOWN_HOURS:
                log.info("%s on cooldown for %.1f more hours", symbol,
                         SYMBOL_COOLDOWN_HOURS - elapsed)
                return True
            else:
                del self.symbol_cooldowns[symbol]
        return False

    def _check_exits(self):
        """Check open positions for trailing stop, take-profit, and stop-loss."""
        positions = self.state.get("positions", {})
        to_close  = []
        state_dirty = False

        for opt_sym, pos in list(positions.items()):
            quote = self.api.get_option_quote(opt_sym)
            if not quote:
                continue
            current = float(quote.get("last") or quote.get("bid") or 0)
            if current <= 0:
                continue
            entry   = pos["entry_price"]
            pnl_pct = (current - entry) / entry * 100

            # ── Trailing Stop Logic ───────────────────────────────────────────
            highest_pnl = pos.get("highest_pnl_pct", pnl_pct)
            if pnl_pct > highest_pnl:
                pos["highest_pnl_pct"] = pnl_pct
                highest_pnl = pnl_pct
                state_dirty = True

            # Activate trailing stop once position is up TRAIL_ACTIVATE_PCT
            if highest_pnl >= TRAIL_ACTIVATE_PCT:
                trail_stop = highest_pnl - TRAIL_DISTANCE_PCT
                if pnl_pct <= trail_stop:
                    to_close.append((opt_sym, pos, current, pnl_pct,
                                     f"TRAILING STOP 🔒 (peak: +{highest_pnl:.1f}%)"))
                    continue

            # ── Fixed Take Profit & Stop Loss ─────────────────────────────────
            if pnl_pct >= TAKE_PROFIT_PCT:
                to_close.append((opt_sym, pos, current, pnl_pct, "TAKE PROFIT ✅"))
            elif pnl_pct <= -STOP_LOSS_PCT:
                to_close.append((opt_sym, pos, current, pnl_pct, "STOP LOSS ❌"))

        # ── Ladder Scaling ────────────────────────────────────────────────────
        if LADDER_MODE:
            for opt_sym, pos in list(positions.items()):
                if opt_sym in [t[0] for t in to_close]:
                    continue  # already closing
                tranche = pos.get("ladder_tranche", 1)
                if tranche >= LADDER_TRANCHES:
                    continue  # fully scaled in
                pnl_pct = pos.get("highest_pnl_pct", 0)
                # Add next tranche when position confirms trend
                if pnl_pct >= LADDER_TRIGGER * tranche:
                    next_tranche  = tranche + 1
                    budget        = MAX_PREMIUM * LADDER_SIZES[next_tranche - 1]
                    quote         = self.api.get_option_quote(opt_sym)
                    if quote:
                        ask = float(quote.get("ask") or quote.get("last") or 0)
                        if ask > 0:
                            add_qty = max(1, int(budget / (ask * 100)))
                            result  = self.api.place_option_order(
                                pos["symbol"], opt_sym, "buy_to_open", add_qty,
                                limit_price=ask
                            )
                            if result:
                                pos["quantity"]       += add_qty
                                pos["ladder_tranche"]  = next_tranche
                                pos["ladder_prices"].append(ask)
                                state_dirty = True
                                send_telegram(
                                    f"📊 LADDER TRANCHE {next_tranche}/{LADDER_TRANCHES}\n"
                                    f"Symbol: {pos['symbol']} {pos['option_type']}\n"
                                    f"Added {add_qty} contract(s) @ ${ask:.2f}\n"
                                    f"Position up {pnl_pct:.1f}% — scaling in ✅"
                                )
                                log.info("Ladder tranche %d — added %d x %s",
                                         next_tranche, add_qty, opt_sym)

        if state_dirty:
            save_state(self.state)

        for opt_sym, pos, price, pnl_pct, reason in to_close:
            # Use bid price for sell limit orders so they fill quickly
            exit_quote = self.api.get_option_quote(opt_sym)
            bid_price  = float((exit_quote or {}).get("bid") or price)
            limit_exit = bid_price if bid_price > 0 else price
            result = self.api.place_option_order(
                pos["symbol"], opt_sym, "sell_to_close", pos["quantity"],
                limit_price=limit_exit
            )
            if result:
                pnl_usd = (price - pos["entry_price"]) * pos["quantity"] * 100
                self.session_pnl += pnl_usd
                send_telegram(
                    f"{'📄 PAPER ' if PAPER_MODE else ''}OPTIONS CLOSE — {reason}\n"
                    f"Contract: {opt_sym}\n"
                    f"Entry: ${pos['entry_price']:.2f}  Exit: ${price:.2f}\n"
                    f"P&L: {pnl_pct:+.1f}% (${pnl_usd:+.2f})\n"
                    f"All-time P&L: ${self.session_pnl:+.2f}"
                )
                entry = {
                    "action":      "close",
                    "reason":      reason,
                    "symbol":      pos["symbol"],
                    "option":      opt_sym,
                    "entry_price": pos["entry_price"],
                    "exit_price":  price,
                    "pnl_pct":     round(pnl_pct, 2),
                    "pnl_usd":     round(pnl_usd, 2),
                    "paper":       PAPER_MODE,
                    "timestamp":   datetime.now().isoformat(),
                }
                log_trade(entry)
                self.daily_trades.append(entry)
                del self.state["positions"][opt_sym]
                save_state(self.state)
                log.info("Closed %s — %s  P&L: %+.1f%% ($%+.2f)",
                         opt_sym, reason, pnl_pct, pnl_usd)

                # Put symbol on cooldown after a loss so it can't re-enter immediately
                if pnl_usd < 0:
                    self.symbol_cooldowns[pos["symbol"]] = datetime.now()
                    log.info("%s placed on %dh cooldown after loss",
                             pos["symbol"], SYMBOL_COOLDOWN_HOURS)

    def _try_entry(self, symbol: str, signal: str,
                   perf: dict = None, straddle: bool = False):
        """Try to enter a new position."""
        # Hard stop — account below minimum or daily loss limit hit
        if self._check_account_minimum():
            return
        if self._check_daily_loss_limit():
            return

        # Symbol on cooldown after recent loss
        if self._symbol_on_cooldown(symbol):
            return

        # Adaptive learning — skip symbols with poor track record
        if perf and not symbol_is_allowed(symbol, perf):
            return

        # ── Risk Filters ──────────────────────────────────────────────────────
        positions = self.state.get("positions", {})
        passed, reason = passes_all_filters(symbol, positions)
        if not passed:
            log.info("❌ Risk filter blocked %s: %s", symbol, reason)
            return

        positions = self.state.get("positions", {})

        # Check if already in this exact direction for this symbol
        for pos in positions.values():
            if pos["symbol"] == symbol and pos["option_type"] == signal:
                log.info("%s %s: already in this position, skipping", symbol, signal)
                return

        if len(positions) >= MAX_POSITIONS:
            log.info("Max positions (%d) reached", MAX_POSITIONS)
            return

        # Get spot price
        quote = self.api.get_quote(symbol)
        if not quote:
            return
        spot = float(quote.get("last") or quote.get("close") or 0)
        if spot <= 0:
            return

        # Pick expiration and option
        expirations = self.api.get_expirations(symbol)
        expiration  = pick_expiration(expirations)
        if not expiration:
            log.info("%s: no expiration in %d–%d DTE window",
                     symbol, TARGET_DTE_MIN, TARGET_DTE_MAX)
            return

        chain  = self.api.get_chain(symbol, expiration)
        option = pick_option(chain, "call" if signal == "CALL" else "put", spot)
        if not option:
            log.info("%s: no suitable %s option found", symbol, signal)
            return

        ask        = float(option.get("ask") or 0)
        opt_symbol = option.get("symbol")
        strike     = float(option.get("strike") or 0)
        greeks     = option.get("greeks") or {}
        delta      = greeks.get("delta", "N/A")
        today_dte  = (datetime.strptime(expiration, "%Y-%m-%d").date() - date.today()).days

        # ── Laddering: Tranche 1 of LADDER_TRANCHES ──────────────────────────
        if LADDER_MODE:
            tranche_budget = MAX_PREMIUM * LADDER_SIZES[0] * vix_premium_multiplier()
            quantity       = max(1, int(tranche_budget / (ask * 100)))
        else:
            quantity       = max(1, int(MAX_PREMIUM / (ask * 100)))

        result = self.api.place_option_order(symbol, opt_symbol, "buy_to_open", quantity,
                                              limit_price=ask)
        if result:
            self.state.setdefault("positions", {})[opt_symbol] = {
                "symbol":         symbol,
                "option_type":    signal,
                "strike":         strike,
                "expiration":     expiration,
                "entry_price":    ask,
                "quantity":       quantity,
                "highest_pnl_pct": 0.0,   # trailing stop tracker
                "ladder_tranche": 1,       # which tranche we're on
                "ladder_prices":  [ask],   # entry prices for each tranche
                "timestamp":      datetime.now().isoformat(),
            }
            save_state(self.state)
            cost = ask * quantity * 100
            ladder_note = f"Tranche 1/{LADDER_TRANCHES} — next at +{LADDER_TRIGGER}%\n" if LADDER_MODE else ""
            send_telegram(
                f"📈 {'PAPER ' if PAPER_MODE else ''}OPTIONS ENTRY\n"
                f"Symbol: {symbol}  Signal: {signal}\n"
                f"Contract: {opt_symbol}\n"
                f"Strike: ${strike:.2f}  Exp: {expiration} ({today_dte} DTE)\n"
                f"Delta: {delta}  Ask: ${ask:.2f}\n"
                f"Qty: {quantity}  Cost: ${cost:.2f}\n"
                f"{ladder_note}"
                f"Trail activates at +{TRAIL_ACTIVATE_PCT}% | SL: -{STOP_LOSS_PCT}%"
            )
            entry = {
                "action":      "open",
                "symbol":      symbol,
                "signal":      signal,
                "option":      opt_symbol,
                "strike":      strike,
                "expiration":  expiration,
                "dte":         today_dte,
                "entry_price": ask,
                "quantity":    quantity,
                "cost":        round(cost, 2),
                "paper":       PAPER_MODE,
                "timestamp":   datetime.now().isoformat(),
            }
            log_trade(entry)
            self.daily_trades.append(entry)
            log.info("Entered %s %s @ $%.2f x%d (cost $%.2f)",
                     signal, opt_symbol, ask, quantity, cost)

    def _run_wheel(self):
        """Execute wheel strategy — sell puts on strong stocks, covered calls if assigned."""
        opportunities = run_wheel_scan(self.api, self.state)
        for symbol, action, option in opportunities:
            try:
                bid        = float(option.get("bid") or 0)
                opt_symbol = option.get("symbol")
                strike     = float(option.get("strike") or 0)
                expiration = option.get("expiration") or option.get("root_symbol", "")
                greeks     = option.get("greeks") or {}
                delta      = greeks.get("delta", "N/A")
                quantity   = 1
                premium    = bid * 100  # premium collected per contract

                # Sell the option
                side = "sell_to_open"
                result = self.api.place_option_order(symbol, opt_symbol, side, quantity,
                                                     limit_price=bid)
                if result:
                    key = f"wheel_{symbol}_{action}"
                    self.state.setdefault("wheel_positions", {})[symbol] = {
                        "symbol":      symbol,
                        "action":      action,
                        "stage":       "put_sold" if action == "sell_put" else "call_sold",
                        "strike":      strike,
                        "expiration":  expiration,
                        "premium":     bid,
                        "quantity":    quantity,
                        "timestamp":   datetime.now().isoformat(),
                    }
                    save_state(self.state)

                    emoji = "🔵" if action == "sell_put" else "🟢"
                    send_telegram(
                        f"{emoji} {'PAPER ' if PAPER_MODE else ''}WHEEL — {action.upper()}\n"
                        f"Symbol: {symbol}\n"
                        f"Strike: ${strike:.2f}  Exp: {expiration}\n"
                        f"Delta: {delta}  Bid: ${bid:.2f}\n"
                        f"Premium collected: ${premium:.2f}\n"
                        f"{'Strategy: collect premium, keep if expires worthless' if action == 'sell_put' else 'Strategy: collect premium while holding shares'}"
                    )
                    log.info("Wheel %s %s strike=%.0f premium=$%.2f",
                             action, symbol, strike, premium)
            except Exception as e:
                log.warning("Wheel %s %s failed: %s", action, symbol, e)

    def _run_spacex_straddle(self):
        """Check for SpaceX IPO proximity and enter straddles on proxy stocks."""
        opportunities = check_spacex_straddle(self.api, self.state)
        for symbol, signal in opportunities:
            self._try_entry(symbol, signal)
            # Mark as spacex position
            self.state.setdefault("spacex_positions", {})[symbol] = {
                "timestamp": datetime.now().isoformat()
            }
            save_state(self.state)

    def _daily_summary(self):
        """Send daily summary at 4 PM with today's P&L and all-time P&L."""
        positions = self.state.get("positions", {})

        # Load full trade log
        all_trades = []
        try:
            with open(TRADE_LOG_FILE) as f:
                all_trades = json.load(f)
        except Exception:
            pass

        all_closes = [t for t in all_trades
                      if t.get("action") == "close" and not t.get("paper", False)]

        # Today's trades
        today_str  = date.today().isoformat()
        today_closes = [t for t in all_closes
                        if t.get("timestamp", "").startswith(today_str)]
        today_opens  = [t for t in all_trades
                        if t.get("action") == "open"
                        and t.get("timestamp", "").startswith(today_str)]

        # P&L calculations
        day_pnl      = sum(t.get("pnl_usd", 0) for t in today_closes)
        alltime_pnl  = sum(t.get("pnl_usd", 0) for t in all_closes)
        all_winners  = [t for t in all_closes if t.get("pnl_usd", 0) > 0]
        all_losers   = [t for t in all_closes if t.get("pnl_usd", 0) <= 0]
        win_rate     = (len(all_winners) / len(all_closes) * 100) if all_closes else 0

        pos_lines = ""
        for opt_sym, pos in positions.items():
            pos_lines += f"  • {pos['symbol']} {pos['option_type']} ${pos['strike']} exp {pos['expiration']}\n"

        send_telegram(
            f"📊 Tradier Daily Summary — {date.today().strftime('%b %d, %Y')}\n"
            f"Mode: {'PAPER 🧪' if PAPER_MODE else 'LIVE 💰'}\n"
            f"─────────────────\n"
            f"TODAY\n"
            f"Trades opened: {len(today_opens)}\n"
            f"Trades closed: {len(today_closes)}\n"
            f"Today's P&L: ${day_pnl:+.2f}\n"
            f"─────────────────\n"
            f"ALL TIME\n"
            f"Total closed trades: {len(all_closes)}\n"
            f"Winners: {len(all_winners)}  Losers: {len(all_losers)}\n"
            f"Win rate: {win_rate:.0f}%\n"
            f"All-time P&L: ${alltime_pnl:+.2f}\n"
            f"─────────────────\n"
            f"Open positions ({len(positions)}/{MAX_POSITIONS}):\n"
            f"{pos_lines if pos_lines else '  None'}"
        )
        self.daily_trades = []
        self.last_summary_day = date.today()

    def run(self):
        log.info("=== Tradier Options Bot STARTED ===")
        balance = self.api.get_account_balance()

        send_telegram(
            f"📈 Tradier Options Bot is online!\n"
            f"Mode: {'PAPER 🧪' if PAPER_MODE else 'LIVE 💰'}\n"
            f"Account balance: ${balance:,.2f}\n"
            f"Max premium per trade: ${MAX_PREMIUM}\n"
            f"Target DTE: {TARGET_DTE_MIN}–{TARGET_DTE_MAX} days\n"
            f"TP: +{TAKE_PROFIT_PCT}%  |  SL: -{STOP_LOSS_PCT}%\n"
            f"Daily loss limit: -${DAILY_LOSS_LIMIT:.0f}\n"
            f"Account minimum: ${ACCOUNT_MINIMUM:,.0f} (halts if breached)\n"
            f"Scanning {len(SYMBOLS)} symbols every {SCAN_INTERVAL//60} min\n"
            f"After-hours: queuing signals for 9:30 AM open 🕙"
        )

        self.signal_queue = {}   # queued signals detected after hours
        self.market_fired = False  # tracks if we fired queued signals at open

        while True:
            try:
                self.scan_count += 1
                now = datetime.now()
                market_open = self._is_market_open(now)

                log.info("=== Scan #%d === market_open=%s ===",
                         self.scan_count, market_open)

                # Check Telegram commands FIRST so /report is never delayed
                check_telegram_commands(self)

                # Fire queued signals at market open (9:30 AM)
                if market_open and not self.market_fired and self.signal_queue:
                    log.info("🔔 Market open — firing %d queued signals",
                             len(self.signal_queue))
                    send_telegram(
                        f"🔔 Market Open — Firing Queued Signals\n"
                        f"Signals queued overnight: {len(self.signal_queue)}\n"
                        f"Symbols: {', '.join(self.signal_queue.keys())}"
                    )
                    for sym, sig in list(self.signal_queue.items()):
                        perf = get_performance_stats()
                        self._try_entry(sym, sig, perf=perf)
                        if STRADDLE_MODE:
                            opposite = "PUT" if sig == "CALL" else "CALL"
                            self._try_entry(sym, opposite, perf=perf, straddle=True)
                    self.signal_queue = {}
                    self.market_fired = True

                # Reset market_fired flag after close so next open fires again
                if not market_open and now.hour >= 16:
                    self.market_fired = False

                # Check exits only during market hours
                if market_open:
                    self._check_exits()

                # Compute adaptive performance stats once per scan
                perf = get_performance_stats()
                rsi_adj = perf.get("rsi_adjustment", 0)
                if rsi_adj > 0:
                    log.info("⚠️ Win rate low — RSI threshold tightened by %dpts", rsi_adj)

                # Get whale signals once per scan (cached hourly)
                whale_sigs = get_whale_signals(lookback_days=3) if POLITICIAN_MODE else []
                if whale_sigs:
                    log.info("🐋 %d whale signals active from top %d politicians",
                             len(whale_sigs), WHALE_TOP_N)

                # Direct copy trades from whales (enter immediately)
                if market_open and POLITICIAN_MODE:
                    for ws in whale_sigs:
                        sym = ws["symbol"]
                        if sym in SYMBOLS or True:  # trade any whale symbol
                            log.info("🐋 Copying whale trade: %s %s by %s (rank #%d)",
                                     ws["signal"], sym, ws["politician"], ws["rank"])
                            self._try_entry(sym, ws["signal"], perf=perf)

                # Scan signals — parallel across all symbols
                check_telegram_commands(self)
                raw_signals = {}

                def scan_symbol(sym):
                    try:
                        pol_adj  = get_politician_signal(sym, whale_sigs)
                        adj      = rsi_adj + pol_adj
                        return sym, get_signal(self.api, sym, rsi_adj=adj)
                    except Exception as e:
                        log.warning("Error scanning %s: %s", sym, e)
                        return sym, None

                with ThreadPoolExecutor(max_workers=8) as executor:
                    futures = {executor.submit(scan_symbol, sym): sym for sym in SYMBOLS}
                    for future in as_completed(futures):
                        sym, signal = future.result()
                        if signal:
                            raw_signals[sym] = signal

                log.info("Scan complete — %d signals found across %d symbols",
                         len(raw_signals), len(SYMBOLS))

                if market_open:
                    # Market is open — execute trades immediately
                    for symbol, signal in raw_signals.items():
                        self._try_entry(symbol, signal, perf=perf)
                        if STRADDLE_MODE:
                            opposite = "PUT" if signal == "CALL" else "CALL"
                            self._try_entry(symbol, opposite, perf=perf, straddle=True)
                        check_telegram_commands(self)
                else:
                    # Market is closed — queue signals for 9:30 AM open
                    new_queued = []
                    for symbol, signal in raw_signals.items():
                        if symbol not in self.signal_queue:
                            self.signal_queue[symbol] = signal
                            new_queued.append(f"{symbol} {signal}")
                    if new_queued:
                        log.info("After hours — queued: %s", ", ".join(new_queued))
                        send_telegram(
                            f"🌙 After-Hours Signal Queue\n"
                            f"New signals queued for market open:\n"
                            f"{chr(10).join('  • ' + s for s in new_queued)}\n"
                            f"Total queued: {len(self.signal_queue)} symbols\n"
                            f"Will fire at 9:30 AM ET"
                        )

                # Reset daily halt at midnight for a new trading day
                now = datetime.now()
                if now.hour == 0 and self.trading_halted:
                    self.trading_halted = False
                    log.info("New day — daily loss limit reset, trading resumed")

                # Wheel strategy — run every scan during market hours
                if market_open and WHEEL_MODE:
                    self._run_wheel()

                # SpaceX IPO straddle — checks proximity to IPO date
                if SPACEX_MODE:
                    self._run_spacex_straddle()

                # Daily summary at 4 PM (market close) — one message per day only
                if (now.hour == DAILY_SUMMARY_HOUR and
                        self.last_summary_day != date.today()):
                    self._daily_summary()

            except KeyboardInterrupt:
                log.info("Bot stopped by user.")
                send_telegram("📈 Tradier Options Bot stopped.")
                break
            except Exception as e:
                log.error("Loop error: %s", e, exc_info=True)
                send_telegram(f"⚠️ Tradier bot error: {e}")

            time.sleep(SCAN_INTERVAL)


# =============================================================================
#  ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    bot = TradierOptionsBot()
    bot.run()
