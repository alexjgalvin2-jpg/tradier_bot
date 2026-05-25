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
import pandas as pd
import numpy as np

# =============================================================================
#  CONFIG
# =============================================================================
TRADIER_TOKEN   = os.getenv("TRADIER_TOKEN", "")
TRADIER_ACCOUNT = os.getenv("TRADIER_ACCOUNT", "")
TELEGRAM_TOKEN  = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

PAPER_MODE = True   # ← set False only when ready for real money

# Tradier sandbox = paper, api = live
BASE_URL = "https://sandbox.tradier.com/v1" if PAPER_MODE else "https://api.tradier.com/v1"

MAX_PREMIUM      = 200.0   # max USD per trade (1 contract = 100 shares)
MAX_POSITIONS    = 6
TAKE_PROFIT_PCT  = 50.0    # exit at +50%
STOP_LOSS_PCT    = 30.0    # exit at -30%
SCAN_INTERVAL    = 300     # 5 minutes between full scans
DAILY_SUMMARY_HOUR = 16    # send daily summary at 4 PM (market close)

TARGET_DTE_MIN   = 20      # widened from 25 — catches more expirations
TARGET_DTE_MAX   = 60      # widened from 50
TARGET_DELTA_MIN = 0.25    # widened from 0.30
TARGET_DELTA_MAX = 0.55    # widened from 0.50

RSI_PERIOD  = 14
RSI_BULL    = 52    # lowered from 55 — triggers more bullish signals
RSI_BEAR    = 48    # raised  from 45 — triggers more bearish signals
SMA_PERIOD  = 20
VOLUME_MULT = 1.1   # lowered from 1.3 — less strict volume spike

# Signal mode: "all" = need price+RSI+volume, "any2" = need 2 of 3 conditions
SIGNAL_MODE = "any2"

SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "TSLA", "META",
    "AMD",  "AMZN", "GOOGL", "SPY",  "QQQ",
    "COIN", "PLTR", "CRWD",  "ARM",  "SHOP",
    "NFLX", "UBER", "SOFI",  "MSTR", "SMCI",
]

STATE_FILE     = "tradier_state.json"
TRADE_LOG_FILE = "tradier_trades.json"

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
def send_telegram(msg: str):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=10)
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
                positions  = bot.state.get("positions", {})
                trade_log  = []
                try:
                    with open(TRADE_LOG_FILE) as f:
                        trade_log = json.load(f)
                except Exception:
                    pass
                closes   = [t for t in trade_log if t.get("action") == "close"]
                total_pnl = sum(t.get("pnl_usd", 0) for t in closes)
                winners  = [t for t in closes if t.get("pnl_usd", 0) > 0]
                win_rate = (len(winners) / len(closes) * 100) if closes else 0

                pos_lines = ""
                for opt_sym, pos in positions.items():
                    pos_lines += f"  • {pos['symbol']} {pos['option_type']} ${pos['strike']} exp {pos['expiration']}\n"

                send_telegram(
                    f"📊 Tradier Report\n"
                    f"Mode: {'PAPER 🧪' if PAPER_MODE else 'LIVE 💰'}\n"
                    f"─────────────────\n"
                    f"Open positions: {len(positions)}/{MAX_POSITIONS}\n"
                    f"{pos_lines if pos_lines else '  None\n'}"
                    f"─────────────────\n"
                    f"Total trades closed: {len(closes)}\n"
                    f"Win rate: {win_rate:.0f}%\n"
                    f"Total P&L: ${total_pnl:+.2f}\n"
                    f"Session P&L: ${bot.session_pnl:+.2f}"
                )
                log.info("Sent /report to Telegram")
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
            # sandbox returns different structure
            cash = (bal.get("cash", {}) or {}).get("cash_available") or \
                   bal.get("total_cash", 0)
            return float(cash or 0)
        return 0.0

    def place_option_order(self, symbol: str, option_symbol: str,
                           side: str, quantity: int) -> Optional[dict]:
        """side: buy_to_open or sell_to_close"""
        if PAPER_MODE:
            log.info("📄 PAPER ORDER: %s %s x%d", side, option_symbol, quantity)
            return {"id": f"paper-{int(time.time())}", "status": "ok", "paper": True}

        data = self._post(f"/accounts/{TRADIER_ACCOUNT}/orders", {
            "class":         "option",
            "symbol":        symbol,
            "option_symbol": option_symbol,
            "side":          side,
            "quantity":      str(quantity),
            "type":          "market",
            "duration":      "day",
        })
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


def get_signal(api: TradierAPI, symbol: str) -> Optional[str]:
    """
    Returns 'CALL', 'PUT', or None.
    SIGNAL_MODE='all'  → needs price + RSI + volume all three
    SIGNAL_MODE='any2' → needs any 2 of the 3 conditions (fires more often)
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

    log.info("%s  price=%.2f  SMA=%.2f  RSI=%.1f  vol_ratio=%.2f",
             symbol, price, sma, rsi, vol_ratio)

    price_bull = price > sma
    price_bear = price < sma
    rsi_bull   = rsi >= RSI_BULL
    rsi_bear   = rsi <= RSI_BEAR
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
        self.session_pnl      = 0.0
        self.daily_trades     = []          # trades opened/closed today
        self.last_summary_day = None        # date of last daily summary
        log.info("TradierOptionsBot initialised — %s mode",
                 "PAPER" if PAPER_MODE else "LIVE")

    def _check_exits(self):
        """Check open positions for take-profit / stop-loss."""
        positions = self.state.get("positions", {})
        to_close  = []

        for opt_sym, pos in list(positions.items()):
            quote = self.api.get_option_quote(opt_sym)
            if not quote:
                continue
            current = float(quote.get("last") or quote.get("bid") or 0)
            if current <= 0:
                continue
            entry   = pos["entry_price"]
            pnl_pct = (current - entry) / entry * 100

            if pnl_pct >= TAKE_PROFIT_PCT:
                to_close.append((opt_sym, pos, current, pnl_pct, "TAKE PROFIT ✅"))
            elif pnl_pct <= -STOP_LOSS_PCT:
                to_close.append((opt_sym, pos, current, pnl_pct, "STOP LOSS ❌"))

        for opt_sym, pos, price, pnl_pct, reason in to_close:
            result = self.api.place_option_order(
                pos["symbol"], opt_sym, "sell_to_close", pos["quantity"]
            )
            if result:
                pnl_usd = (price - pos["entry_price"]) * pos["quantity"] * 100
                self.session_pnl += pnl_usd
                send_telegram(
                    f"{'📄 PAPER ' if PAPER_MODE else ''}OPTIONS CLOSE — {reason}\n"
                    f"Contract: {opt_sym}\n"
                    f"Entry: ${pos['entry_price']:.2f}  Exit: ${price:.2f}\n"
                    f"P&L: {pnl_pct:+.1f}% (${pnl_usd:+.2f})\n"
                    f"Session P&L: ${self.session_pnl:+.2f}"
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

    def _try_entry(self, symbol: str, signal: str):
        """Try to enter a new position."""
        positions = self.state.get("positions", {})

        # Already in a trade for this symbol?
        for pos in positions.values():
            if pos["symbol"] == symbol:
                log.info("%s: already in a position, skipping", symbol)
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
        quantity   = max(1, int(MAX_PREMIUM / (ask * 100)))
        today_dte  = (datetime.strptime(expiration, "%Y-%m-%d").date() - date.today()).days

        result = self.api.place_option_order(symbol, opt_symbol, "buy_to_open", quantity)
        if result:
            self.state.setdefault("positions", {})[opt_symbol] = {
                "symbol":      symbol,
                "option_type": signal,
                "strike":      strike,
                "expiration":  expiration,
                "entry_price": ask,
                "quantity":    quantity,
                "timestamp":   datetime.now().isoformat(),
            }
            save_state(self.state)
            cost = ask * quantity * 100
            send_telegram(
                f"📈 {'PAPER ' if PAPER_MODE else ''}OPTIONS ENTRY\n"
                f"Symbol: {symbol}  Signal: {signal}\n"
                f"Contract: {opt_symbol}\n"
                f"Strike: ${strike:.2f}  Exp: {expiration} ({today_dte} DTE)\n"
                f"Delta: {delta}  Ask: ${ask:.2f}\n"
                f"Qty: {quantity}  Cost: ${cost:.2f}\n"
                f"TP: +{TAKE_PROFIT_PCT}%  SL: -{STOP_LOSS_PCT}%"
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

    def _daily_summary(self):
        """Send one summary message at market close (4 PM). No other periodic alerts."""
        positions = self.state.get("positions", {})
        today_trades = [t for t in self.daily_trades]

        opens  = [t for t in today_trades if t["action"] == "open"]
        closes = [t for t in today_trades if t["action"] == "close"]
        day_pnl = sum(t.get("pnl_usd", 0) for t in closes)

        pos_lines = ""
        for opt_sym, pos in positions.items():
            pos_lines += f"  • {pos['symbol']} {pos['option_type']} ${pos['strike']} exp {pos['expiration']}\n"

        send_telegram(
            f"📊 Tradier Daily Summary — {date.today().strftime('%b %d')}\n"
            f"Mode: {'PAPER 🧪' if PAPER_MODE else 'LIVE 💰'}\n"
            f"─────────────────\n"
            f"Trades opened today: {len(opens)}\n"
            f"Trades closed today: {len(closes)}\n"
            f"Today's P&L: ${day_pnl:+.2f}\n"
            f"Session total P&L: ${self.session_pnl:+.2f}\n"
            f"─────────────────\n"
            f"Open positions ({len(positions)}/{MAX_POSITIONS}):\n"
            f"{pos_lines if pos_lines else '  None'}"
        )
        self.daily_trades = []   # reset for next day
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
            f"Scanning {len(SYMBOLS)} symbols every {SCAN_INTERVAL//60} min"
        )
        while True:
            try:
                self.scan_count += 1
                log.info("=== Scan #%d ===", self.scan_count)

                # Check exits first
                self._check_exits()

                # Scan for entries
                for symbol in SYMBOLS:
                    try:
                        signal = get_signal(self.api, symbol)
                        if signal:
                            log.info("%s: %s signal — looking for contract", symbol, signal)
                            self._try_entry(symbol, signal)
                    except Exception as e:
                        log.warning("Error scanning %s: %s", symbol, e)
                    time.sleep(1)  # gentle rate limiting

                # Check for /report command from Telegram
                check_telegram_commands(self)

                # Daily summary at 4 PM (market close) — one message per day only
                now = datetime.now()
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
