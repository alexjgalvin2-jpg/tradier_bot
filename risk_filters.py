"""
Risk Filters — shared by Tradier and IB bots
============================================
1. VIX Filter          — adjusts position size based on market fear level
2. Earnings Blackout   — skips stocks within 5 days of earnings (IV crush risk)
3. Sector Correlation  — limits how many positions per sector at once
4. Conviction Sizing   — bigger positions on stronger signals
5. Time of Day Filter  — only trade first/last hour when liquidity is highest
6. Intraday Confirmation — verify 15-min trend before entering
"""

import logging
import time
from datetime import datetime, date, timedelta
from typing import Optional
from zoneinfo import ZoneInfo
import yfinance as yf

log = logging.getLogger("RiskFilters")

EASTERN = ZoneInfo("America/New_York")  # servers run UTC — trading-time checks must use ET

# =============================================================================
#  VIX FILTER
# =============================================================================
VIX_LOW       = 20.0   # below this = cheap options, trade normally
VIX_HIGH      = 30.0   # above this = expensive options, reduce/stop buying
VIX_CACHE_TTL = 900    # refresh VIX every 15 minutes

_vix_cache     = None
_vix_cache_time = 0


def get_vix() -> float:
    """Returns current VIX level. Cached for 15 minutes."""
    global _vix_cache, _vix_cache_time
    now = time.time()
    if _vix_cache is not None and (now - _vix_cache_time) < VIX_CACHE_TTL:
        return _vix_cache
    try:
        hist = yf.Ticker("^VIX").history(period="2d", interval="1d")
        if hist is not None and not hist.empty:
            vix = float(hist["Close"].iloc[-1])
            _vix_cache      = vix
            _vix_cache_time = now
            log.info("VIX: %.1f", vix)
            return vix
    except Exception as e:
        log.warning("VIX fetch failed: %s", e)
    return _vix_cache if _vix_cache else 20.0  # safe default


def get_vix_regime() -> str:
    """
    Returns the current market regime based on VIX:
      'low'    → VIX < 20, options cheap, trade normally
      'medium' → VIX 20-30, options expensive, reduce size by 50%
      'high'   → VIX > 30, do NOT buy options, wheel/sell premium only
    """
    vix = get_vix()
    if vix < VIX_LOW:
        return "low"
    elif vix < VIX_HIGH:
        return "medium"
    else:
        return "high"


def vix_premium_multiplier() -> float:
    """
    Returns a multiplier for MAX_PREMIUM based on VIX:
      low    → 1.0  (full size)
      medium → 0.5  (half size)
      high   → 0.0  (don't buy long options)
    """
    regime = get_vix_regime()
    if regime == "low":
        return 1.0
    elif regime == "medium":
        log.info("⚠️ VIX elevated — reducing position size by 50%%")
        return 0.5
    else:
        log.warning("🚨 VIX > 30 — long options too expensive, skipping entry")
        return 0.0


def vix_status_text() -> str:
    vix    = get_vix()
    regime = get_vix_regime()
    emoji  = "🟢" if regime == "low" else "🟡" if regime == "medium" else "🔴"
    advice = {
        "low":    "Cheap options — full size trading",
        "medium": "Elevated fear — half size trading",
        "high":   "High fear — wheel strategy only",
    }
    return f"{emoji} VIX: {vix:.1f} ({advice[regime]})"


# =============================================================================
#  EARNINGS BLACKOUT
# =============================================================================
EARNINGS_BLACKOUT_DAYS = 5   # skip if earnings within this many days
_earnings_cache = {}          # symbol → (earnings_date, cache_timestamp)
EARNINGS_CACHE_TTL = 86400    # refresh earnings dates once per day


def get_next_earnings(symbol: str) -> Optional[date]:
    """
    Returns the next earnings date for a symbol, or None if unknown.
    Cached per symbol for 24 hours.
    """
    now = time.time()
    if symbol in _earnings_cache:
        cached_date, cached_time = _earnings_cache[symbol]
        if (now - cached_time) < EARNINGS_CACHE_TTL:
            return cached_date

    try:
        tkr = yf.Ticker(symbol)

        # Try earnings_dates (newer yfinance)
        try:
            ed = tkr.earnings_dates
            if ed is not None and not ed.empty:
                today = date.today()
                future = [d.date() for d in ed.index if d.date() >= today]
                if future:
                    result = min(future)
                    _earnings_cache[symbol] = (result, now)
                    return result
        except Exception:
            pass

        # Try calendar (older yfinance)
        try:
            cal = tkr.calendar
            if cal is not None and not cal.empty:
                earnings_date = cal.columns[0]
                if hasattr(earnings_date, 'date'):
                    result = earnings_date.date()
                    _earnings_cache[symbol] = (result, now)
                    return result
        except Exception:
            pass

    except Exception as e:
        log.debug("get_next_earnings %s: %s", symbol, e)

    _earnings_cache[symbol] = (None, now)
    return None


def is_earnings_blackout(symbol: str) -> bool:
    """
    Returns True if the stock has earnings within EARNINGS_BLACKOUT_DAYS.
    We skip these to avoid IV crush after earnings.
    """
    earnings = get_next_earnings(symbol)
    if earnings is None:
        return False

    today     = date.today()
    days_away = (earnings - today).days

    if 0 <= days_away <= EARNINGS_BLACKOUT_DAYS:
        log.info("🚫 %s earnings in %d days (%s) — skipping to avoid IV crush",
                 symbol, days_away, earnings)
        return True
    return False


def earnings_status_text(symbol: str) -> str:
    earnings = get_next_earnings(symbol)
    if earnings is None:
        return f"{symbol}: earnings date unknown"
    days_away = (earnings - date.today()).days
    emoji = "🚫" if days_away <= EARNINGS_BLACKOUT_DAYS else "✅"
    return f"{emoji} {symbol}: earnings in {days_away} days ({earnings})"


# =============================================================================
#  SECTOR CORRELATION LIMIT
# =============================================================================
SECTORS = {
    "tech":     ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMD",
                 "INTC", "QCOM", "ARM",  "SMCI"],
    "growth":   ["TSLA", "SHOP", "COIN", "PLTR", "CRWD", "HOOD",
                 "SOFI", "MSTR", "RBLX", "DKNG", "SNAP"],
    "consumer": ["NFLX", "UBER", "AMZN"],
    "etf":      ["SPY",  "QQQ",  "IWM",  "ARKK", "SOXL"],
    "finance":  ["JPM",  "GS",   "BAC",  "MS"],
    "energy":   ["XOM",  "CVX",  "OXY"],
    "pharma":   ["LLY",  "MRNA", "BNTX"],
    "space":    ["RKLB", "ASTS", "BA",   "LMT"],
    "crypto":   ["COIN", "MSTR", "HOOD"],
}

MAX_SECTOR_POSITIONS = 3  # max open positions per sector at once


def get_sector(symbol: str) -> str:
    """Returns the sector for a given symbol."""
    for sector, symbols in SECTORS.items():
        if symbol in symbols:
            return sector
    return "other"


def sector_at_limit(symbol: str, positions: dict) -> bool:
    """
    Returns True if we already have MAX_SECTOR_POSITIONS open
    in the same sector as this symbol.
    """
    sector      = get_sector(symbol)
    sector_count = sum(
        1 for pos in positions.values()
        if get_sector(pos.get("symbol", "")) == sector
    )
    if sector_count >= MAX_SECTOR_POSITIONS:
        log.info("📊 %s sector '%s' at limit (%d/%d) — skipping",
                 symbol, sector, sector_count, MAX_SECTOR_POSITIONS)
        return True
    return False


def sector_exposure_text(positions: dict) -> str:
    """Returns a summary of current sector exposure."""
    counts = {}
    for pos in positions.values():
        sector = get_sector(pos.get("symbol", "other"))
        counts[sector] = counts.get(sector, 0) + 1

    if not counts:
        return "No open positions"

    lines = ""
    for sector, count in sorted(counts.items(), key=lambda x: -x[1]):
        bar   = "█" * count
        warn  = " ⚠️" if count >= MAX_SECTOR_POSITIONS else ""
        lines += f"  {sector}: {bar} {count}{warn}\n"
    return lines


# =============================================================================
#  TIME OF DAY FILTER
# =============================================================================
TRADE_WINDOWS = [
    (9, 30, 11, 0),   # Morning: 9:30 AM - 11:00 AM (best liquidity)
    (15, 0, 16, 0),   # Closing: 3:00 PM - 4:00 PM  (second best)
]


def is_good_trading_time() -> bool:
    """
    Returns True if current time is within optimal trading windows.
    Avoids 11 AM - 3 PM dead zone where spreads are widest.
    """
    now = datetime.now(EASTERN)
    if now.weekday() >= 5:
        return False
    for h_start, m_start, h_end, m_end in TRADE_WINDOWS:
        window_start = now.replace(hour=h_start, minute=m_start, second=0)
        window_end   = now.replace(hour=h_end,   minute=m_end,   second=0)
        if window_start <= now <= window_end:
            return True
    log.debug("Outside trading window — skipping entry (best times: 9:30-11am, 3-4pm)")
    return False


def trading_time_status() -> str:
    now = datetime.now(EASTERN)
    h, m = now.hour, now.minute
    if is_good_trading_time():
        return f"✅ Good trading time ({h:02d}:{m:02d})"
    return f"⏳ Outside optimal window ({h:02d}:{m:02d}) — waiting for 9:30-11am or 3-4pm"


# =============================================================================
#  CONVICTION-BASED POSITION SIZING
# =============================================================================
def conviction_multiplier(rsi: float, vol_ratio: float, signal: str) -> float:
    """
    Returns a position size multiplier (0.5 to 1.5) based on signal strength.
    Stronger signals get bigger positions.

    RSI 55-65 + vol 1.3-1.5x → 0.6x (cautious)
    RSI 65-75 + vol 1.5-2.5x → 1.0x (normal)
    RSI 75+   + vol 2.5x+    → 1.5x (high conviction)
    """
    score = 0

    if signal == "CALL":
        if rsi >= 80:   score += 3
        elif rsi >= 70: score += 2
        elif rsi >= 60: score += 1
    else:  # PUT
        rsi_inv = 100 - rsi
        if rsi_inv >= 80:   score += 3
        elif rsi_inv >= 70: score += 2
        elif rsi_inv >= 60: score += 1

    if vol_ratio >= 3.0:   score += 3
    elif vol_ratio >= 2.0: score += 2
    elif vol_ratio >= 1.5: score += 1

    if score >= 5:   return 1.5   # high conviction — 150% size
    elif score >= 3: return 1.0   # normal
    else:            return 0.6   # cautious — 60% size


# =============================================================================
#  INTRADAY CONFIRMATION (15-min chart)
# =============================================================================
_intraday_cache     = {}
_intraday_cache_time = {}
INTRADAY_CACHE_TTL  = 900  # 15 minutes


def intraday_confirms(symbol: str, signal: str) -> bool:
    """
    Checks 15-minute chart to confirm daily signal direction.
    Prevents entering calls when intraday trend is already turning down.

    Returns True if intraday momentum agrees with the signal.
    """
    now = time.time()
    cache_key = f"{symbol}_{signal}"
    if cache_key in _intraday_cache:
        if (now - _intraday_cache_time.get(cache_key, 0)) < INTRADAY_CACHE_TTL:
            return _intraday_cache[cache_key]
    try:
        tkr  = yf.Ticker(symbol)
        hist = tkr.history(period="1d", interval="15m", auto_adjust=True)
        if hist is None or len(hist) < 5:
            return True  # no data → don't block the trade

        close  = hist["Close"].squeeze()
        # Short EMA vs longer EMA on 15-min bars
        ema5   = close.ewm(span=5).mean().iloc[-1]
        ema15  = close.ewm(span=15).mean().iloc[-1]
        recent_rsi_val = 0
        delta = close.diff().dropna()
        gain  = delta.clip(lower=0)
        loss  = (-delta).clip(lower=0)
        avg_g = gain.ewm(com=6, min_periods=7).mean().iloc[-1]
        avg_l = loss.ewm(com=6, min_periods=7).mean().iloc[-1]
        if avg_l > 0:
            recent_rsi_val = 100 - (100 / (1 + avg_g / avg_l))

        if signal == "CALL":
            # Confirm: short EMA above long EMA and intraday RSI > 45
            confirmed = ema5 > ema15 and recent_rsi_val > 45
        else:
            # Confirm: short EMA below long EMA and intraday RSI < 55
            confirmed = ema5 < ema15 and recent_rsi_val < 55

        _intraday_cache[cache_key]      = confirmed
        _intraday_cache_time[cache_key] = now

        if not confirmed:
            log.info("📉 %s: intraday trend doesn't confirm %s signal (ema5=%.2f ema15=%.2f rsi=%.1f)",
                     symbol, signal, ema5, ema15, recent_rsi_val)
        return confirmed

    except Exception as e:
        log.debug("intraday_confirms %s: %s", symbol, e)
        return True  # on error, don't block


# =============================================================================
#  COMBINED RISK CHECK
# =============================================================================
def passes_all_filters(symbol: str, positions: dict,
                       signal: str = None,
                       rsi: float = None,
                       vol_ratio: float = None,
                       check_intraday: bool = True,
                       check_time: bool = True) -> tuple:
    """
    Run all risk filters. Returns (passed: bool, reason: str).
    """
    # 1. VIX check
    multiplier = vix_premium_multiplier()
    if multiplier == 0.0:
        return False, f"VIX too high ({get_vix():.1f}) — not buying options"

    # 2. Earnings blackout
    if is_earnings_blackout(symbol):
        earnings = get_next_earnings(symbol)
        return False, f"Earnings blackout ({earnings})"

    # 3. Sector limit
    if sector_at_limit(symbol, positions):
        sector = get_sector(symbol)
        return False, f"Sector '{sector}' at limit ({MAX_SECTOR_POSITIONS} positions)"

    # 4. Time of day (only during market hours trading windows)
    if check_time and not is_good_trading_time():
        return False, "Outside optimal trading window"

    # 5. Intraday confirmation
    if check_intraday and signal:
        if not intraday_confirms(symbol, signal):
            return False, f"Intraday trend doesn't confirm {signal}"

    return True, "ok"
