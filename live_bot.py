import json
import time
from dataclasses import dataclass
from datetime import datetime, date
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import ta
import MetaTrader5 as mt5

from mt5_exec import (
    ExecConfig,
    ensure_initialized,
    ensure_symbol,
    account_snapshot,
    get_position,
    close_position_market,
    open_long_by_notional,
)

# ==========================
# CONFIG
# ==========================

STATE_FILE = "live_state.json"

# --- Symbols ---
# Use EXACT MT5 names:
MARKETS = [
    {"name": "US500", "symbol": "US500.cash"},
    {"name": "US100", "symbol": "US100.cash"},
    {"name": "US30",  "symbol": "US30.cash"},
]

# --- Portfolio / exposure ---
MAX_GROSS_EXPOSURE_TOTAL = 2.0

# Strategy market weights (fixed, per your plan)
TF_MARKET_WEIGHTS = {"US500": 0.34, "US100": 0.29, "US30": 0.37}
MR_MARKET_WEIGHTS = {"US500": 1/3,  "US100": 1/3,  "US30": 1/3}

STRATEGY_MARKET_WEIGHTS = {
    "TF": TF_MARKET_WEIGHTS,
    "MR": MR_MARKET_WEIGHTS,
}

# Base strategy exposure budgets (inside total cap)
STRATEGY_TARGETS_BASE = {
    "TF": 1.0,
    "MR": 1.0,
}

# Mode: "EVAL" or "FUNDED"
MODE = "EVAL"

EVAL_EXPOSURE_FACTOR = 1.25
FUNDED_EXPOSURE_FACTOR = 1.00

# --- FTMO risk gates (2-step) ---
# Soft cutoff is relative to DAY START equity
SOFT_CUTOFF_DAILY = -0.04  # -4%: stop new entries rest of day

# Hard limits (absolute) based on initial balance (FTMO-style)
START_BALANCE_FOR_LIMITS = 50_000  # set to your account initial balance
DAILY_LOSS_LIMIT_ABS_FRAC = 0.05   # 5% daily loss (absolute)
MAX_LOSS_LIMIT_ABS_FRAC = 0.10     # 10% max loss (absolute)

# --- Strategy params (match research) ---
TF_PARAMS = dict(exit_confirm_bars=10, adx_threshold=15, ema_fast_len=70, ema_slow_len=120)
MR_PARAMS = dict(ema_fast_len=20, ema_slow_len=250, pullback_frac=0.20)

# --- Data windows ---
H1_BARS = 600
D1_BARS = 800

# --- Polling ---
SLEEP_SECONDS = 1.0
HEARTBEAT_EVERY_SEC = 60

# If you want to hard-disable trading outside hours, implement below.
ENABLE_SESSION_FILTER = False


# ==========================
# STATE
# ==========================

def load_state() -> dict:
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_state(state: dict) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ==========================
# MT5 DATA HELPERS
# ==========================

def mt5_timeframe(tf: str):
    if tf == "H1":
        return mt5.TIMEFRAME_H1
    if tf == "D1":
        return mt5.TIMEFRAME_D1
    raise ValueError("Unsupported timeframe")


_last_rate_error_log = {}  # (symbol, tf) -> last_log_epoch
def fetch_ohlc(symbol: str, tf: str, n: int, min_bars: int = 120) -> pd.DataFrame:
    """
    Robust OHLC fetch:
      - tries to select symbol
      - retries a couple times
      - if still too few bars, returns EMPTY df (so caller can skip)
      - throttles error logs
    """
    ensure_symbol(symbol)

    tf_mt5 = mt5_timeframe(tf)

    # Try a few times (MT5 can intermittently return None)
    for attempt in range(3):
        rates = mt5.copy_rates_from_pos(symbol, tf_mt5, 0, n)
        if rates is not None and len(rates) >= min_bars:
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df = df.set_index("time")
            return df[["open", "high", "low", "close", "tick_volume"]].copy()
        time.sleep(0.2 * (attempt + 1))

    # Still not enough data => return empty and throttle log
    now = int(time.time())
    key = (symbol, tf)
    last = _last_rate_error_log.get(key, 0)
    if now - last >= 60:
        _last_rate_error_log[key] = now
        got = 0 if rates is None else len(rates)
        print(f"[{now_str()}] WARN: No/too few rates for {symbol} {tf} (got {got}, need >= {min_bars}). Skipping.")
    return pd.DataFrame()

def current_bar_open_time(symbol: str, tf: str) -> Optional[pd.Timestamp]:
    df = fetch_ohlc(symbol, tf, 10, min_bars=2)
    if df.empty or len(df) < 2:
        return None
    return pd.Timestamp(df.index[-1])

def last_closed_bar(df: pd.DataFrame) -> pd.Series:
    """
    MT5 rates include current forming bar at the end. Last closed is -2.
    """
    return df.iloc[-2]

def prev_closed_bar(df: pd.DataFrame) -> pd.Series:
    """
    Previous closed bar is -3.
    """
    return df.iloc[-3]


# ==========================
# MAGIC NUMBERS (one per market+strategy)
# ==========================

def magic_for(market_name: str, strategy: str) -> int:
    base = 11000 if strategy == "TF" else 21000
    h = abs(hash(market_name)) % 1000
    return base + h


# ==========================
# STRATEGY SIGNALS (CLOSED BAR)
# ==========================

def compute_tf_indicators(df_h1: pd.DataFrame) -> pd.DataFrame:
    df = df_h1.copy()
    df["ema_fast"] = df["close"].ewm(span=TF_PARAMS["ema_fast_len"], adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=TF_PARAMS["ema_slow_len"], adjust=False).mean()
    df["adx"] = ta.trend.ADXIndicator(
        high=df["high"], low=df["low"], close=df["close"],
        window=14, fillna=False
    ).adx()
    return df

def tf_entry_signal(prev: pd.Series, prev2: pd.Series) -> bool:
    if np.isnan(prev["adx"]) or np.isnan(prev["ema_slow"]) or np.isnan(prev2["ema_slow"]):
        return False
    adx_ok = float(prev["adx"]) > float(TF_PARAMS["adx_threshold"])
    cross_up = (float(prev2["ema_fast"]) < float(prev2["ema_slow"])) and (float(prev["ema_fast"]) > float(prev["ema_slow"]))
    return bool(adx_ok and cross_up)

def tf_exit_breach(prev: pd.Series) -> bool:
    # breach if ema_fast <= ema_slow on closed bar
    if np.isnan(prev["ema_fast"]) or np.isnan(prev["ema_slow"]):
        return False
    return bool(float(prev["ema_fast"]) <= float(prev["ema_slow"]))

def compute_mr_indicators(df_d1: pd.DataFrame) -> pd.DataFrame:
    df = df_d1.copy()
    df["ema_fast"] = df["close"].ewm(span=MR_PARAMS["ema_fast_len"], adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=MR_PARAMS["ema_slow_len"], adjust=False).mean()
    return df

def mr_entry_signal(prev_day: pd.Series) -> bool:
    if np.isnan(prev_day["ema_fast"]) or np.isnan(prev_day["ema_slow"]):
        return False
    close_px = float(prev_day["close"])
    ema_fast = float(prev_day["ema_fast"])
    ema_slow = float(prev_day["ema_slow"])
    high = float(prev_day["high"])
    low = float(prev_day["low"])

    bullish_pullback_regime = (close_px < ema_fast) and (close_px > ema_slow)
    deep_pullback = close_px < (low + float(MR_PARAMS["pullback_frac"]) * (high - low))
    return bool(bullish_pullback_regime and deep_pullback)

def mr_exit_signal(prev_day: pd.Series) -> bool:
    # exit trigger: high >= ema_fast on last closed day
    if np.isnan(prev_day["ema_fast"]):
        return False
    return bool(float(prev_day["high"]) >= float(prev_day["ema_fast"]))


# ==========================
# RISK GATES (2-step style)
# ==========================

def strategy_targets_for_mode() -> Dict[str, float]:
    factor = EVAL_EXPOSURE_FACTOR if MODE.upper() == "EVAL" else FUNDED_EXPOSURE_FACTOR
    return {k: float(v) * float(factor) for k, v in STRATEGY_TARGETS_BASE.items()}

def day_drawdown_frac(day_start_equity: float, equity_now: float) -> float:
    if day_start_equity <= 0:
        return 0.0
    return float(equity_now / day_start_equity - 1.0)

def check_risk_gates(state: dict, equity_now: float) -> Tuple[bool, str]:
    """
    Returns (allow_new_entries, reason)
    - soft cutoff uses intraday dd from day-start equity
    - hard daily/max loss use absolute lines based on START_BALANCE_FOR_LIMITS
    """
    day_start = float(state.get("day_start_equity", equity_now))
    dd_day = day_drawdown_frac(day_start, equity_now)

    # soft cutoff => no new entries rest of day
    if dd_day <= float(SOFT_CUTOFF_DAILY):
        return False, f"SOFT_CUTOFF hit dd_day={dd_day:.2%}"

    # hard daily loss: compare equity to (day_start - 5% of initial)
    daily_loss_abs = float(DAILY_LOSS_LIMIT_ABS_FRAC) * float(START_BALANCE_FOR_LIMITS)
    daily_floor = day_start - daily_loss_abs
    if equity_now < daily_floor:
        return False, "HARD MaxDailyLoss breached"

    # hard max loss: equity < initial*(1-10%)
    max_floor = float(START_BALANCE_FOR_LIMITS) * (1.0 - float(MAX_LOSS_LIMIT_ABS_FRAC))
    if equity_now < max_floor:
        return False, "HARD MaxLoss breached"

    return True, "OK"


# ==========================
# EXPOSURE / CAPACITY
# ==========================

def mid_price(symbol: str) -> float:
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return 0.0
    return (float(tick.bid) + float(tick.ask)) / 2.0

def open_positions_gross_notional() -> float:
    """
    Gross notional proxy consistent with your backtest model:
      notional = price * volume * contract_size
    For your symbols: contract_size=1, so notional ≈ price * volume
    """
    gross = 0.0
    pos = mt5.positions_get()
    if pos is None:
        return 0.0
    for p in pos:
        info = mt5.symbol_info(p.symbol)
        if info is None:
            continue
        cs = float(info.trade_contract_size) if info.trade_contract_size else 1.0
        gross += abs(float(p.price_current) * float(p.volume) * cs)
    return float(gross)

def remaining_gross_capacity(equity_now: float) -> float:
    gross = open_positions_gross_notional()
    cap = float(equity_now) * float(MAX_GROSS_EXPOSURE_TOTAL)
    return float(max(0.0, cap - gross))

def desired_notional(equity_now: float, strat: str, market_name: str) -> float:
    targets = strategy_targets_for_mode()
    w = float(STRATEGY_MARKET_WEIGHTS[strat][market_name])
    t = float(targets.get(strat, 0.0))
    return float(equity_now * t * w)


# ==========================
# BAR SYNC
# ==========================

def is_new_bar(state: dict, tf: str, clock_symbol: str) -> Tuple[bool, Optional[pd.Timestamp]]:
    key = f"last_{tf.lower()}_bar"
    last = state.get(key, None)
    last_ts = pd.to_datetime(last) if last else None

    cur = current_bar_open_time(clock_symbol, tf)
    if cur is None:
        return False, None

    if last_ts is None or cur > last_ts:
        state[key] = str(cur)
        return True, cur

    return False, cur


# ==========================
# LIVE ACTIONS (ENTRY/EXIT)
# ==========================

def run_tf_on_new_h1(cfg: ExecConfig, state: dict, allow_entries: bool) -> None:
    snap = account_snapshot()
    equity_now = float(snap["equity"])
    rem_total = remaining_gross_capacity(equity_now)

    for m in MARKETS:
        mkt = m["name"]
        sym = m["symbol"]
        ensure_symbol(sym)

        magic = magic_for(mkt, "TF")
        pos = get_position(sym, magic)

        df = fetch_ohlc(sym, "H1", H1_BARS, min_bars=200)
        if df.empty:
            continue
        df = compute_tf_indicators(df)

        prev = last_closed_bar(df)
        prev2 = prev_closed_bar(df)

        # --- Exit management (delayed recross confirm) ---
        bc_map = state.setdefault("tf_exit_breach_count", {})
        bc = int(bc_map.get(mkt, 0))

        if pos is not None:
            if tf_exit_breach(prev):
                bc += 1
            else:
                bc = 0

            bc_map[mkt] = bc
            save_state(state)

            if bc >= int(TF_PARAMS["exit_confirm_bars"]):
                close_position_market(sym, magic, cfg, comment="TF_EXIT")
                bc_map[mkt] = 0
                save_state(state)
            continue  # no entry while in position

        # --- Entry ---
        if not allow_entries:
            continue

        if rem_total <= 0:
            continue

        if tf_entry_signal(prev, prev2):
            dn = desired_notional(equity_now, "TF", mkt)
            dn = min(dn, rem_total)
            ok, vol = open_long_by_notional(sym, dn, magic, cfg, comment="TF_ENTRY")
            if ok:
                # reduce remaining capacity by intended notional (conservative)
                rem_total = max(0.0, rem_total - dn)


def run_mr_on_new_d1(cfg: ExecConfig, state: dict, allow_entries: bool) -> None:
    snap = account_snapshot()
    equity_now = float(snap["equity"])
    rem_total = remaining_gross_capacity(equity_now)

    for m in MARKETS:
        mkt = m["name"]
        sym = m["symbol"]
        ensure_symbol(sym)

        magic = magic_for(mkt, "MR")
        pos = get_position(sym, magic)

        df = fetch_ohlc(sym, "D1", D1_BARS, min_bars=300)
        if df.empty:
            continue
        df = compute_mr_indicators(df)
        prev_day = last_closed_bar(df)

        # --- Exit ---
        if pos is not None:
            if mr_exit_signal(prev_day):
                close_position_market(sym, magic, cfg, comment="MR_EXIT")
            continue

        # --- Entry ---
        if not allow_entries:
            continue
        if rem_total <= 0:
            continue

        if mr_entry_signal(prev_day):
            dn = desired_notional(equity_now, "MR", mkt)
            dn = min(dn, rem_total)
            ok, vol = open_long_by_notional(sym, dn, magic, cfg, comment="MR_ENTRY")
            if ok:
                rem_total = max(0.0, rem_total - dn)


# ==========================
# MAIN LOOP
# ==========================

def main():
    ensure_initialized()
    cfg = ExecConfig(log_csv_path="trade_log.csv", retries=3)

    print(f"[{now_str()}] Warming up symbols/history...")
    for m in MARKETS:
        ensure_symbol(m["symbol"])
        _ = mt5.copy_rates_from_pos(m["symbol"], mt5.TIMEFRAME_H1, 0, 5)
        _ = mt5.copy_rates_from_pos(m["symbol"], mt5.TIMEFRAME_D1, 0, 5)
    time.sleep(1.0)
    print(f"[{now_str()}] Warmup done.")

    # State init
    state = load_state()
    state.setdefault("last_h1_bar", None)
    state.setdefault("last_d1_bar", None)
    state.setdefault("day_start_date", None)
    state.setdefault("day_start_equity", None)
    state.setdefault("tf_exit_breach_count", {})
    save_state(state)

    # Use first symbol as bar clock
    clock_symbol = MARKETS[0]["symbol"]

    print(f"[{now_str()}] LIVE BOT STARTED | MODE={MODE} | EvalFactor={EVAL_EXPOSURE_FACTOR:.2f}")
    print(f"Symbols: {[m['symbol'] for m in MARKETS]}")
    print(f"MAX_GROSS_EXPOSURE_TOTAL={MAX_GROSS_EXPOSURE_TOTAL:.2f}")
    print(f"SOFT_CUTOFF_DAILY={SOFT_CUTOFF_DAILY:.2%} | HardDailyAbs={DAILY_LOSS_LIMIT_ABS_FRAC:.2%} | HardMaxAbs={MAX_LOSS_LIMIT_ABS_FRAC:.2%}")
    print("Strategy targets (base):", STRATEGY_TARGETS_BASE)
    print("Strategy market weights:", STRATEGY_MARKET_WEIGHTS)

    last_heartbeat = 0

    while True:
        try:
            snap = account_snapshot()
            equity_now = float(snap["equity"])

            # Day reset
            today = date.today().isoformat()
            if state.get("day_start_date") != today:
                state["day_start_date"] = today
                state["day_start_equity"] = equity_now
                # reset per-day things if you want
                save_state(state)
                print(f"[{now_str()}] New day: day_start_equity={equity_now:.2f}")

            allow_entries, reason = check_risk_gates(state, equity_now)

            # Bar triggers
            new_h1, h1_time = is_new_bar(state, "H1", clock_symbol)
            if new_h1:
                save_state(state)
                run_tf_on_new_h1(cfg, state, allow_entries)

            new_d1, d1_time = is_new_bar(state, "D1", clock_symbol)
            if new_d1:
                save_state(state)
                run_mr_on_new_d1(cfg, state, allow_entries)

            # Heartbeat
            now_sec = int(time.time())
            if now_sec - last_heartbeat >= HEARTBEAT_EVERY_SEC:
                last_heartbeat = now_sec
                dd = day_drawdown_frac(float(state.get("day_start_equity", equity_now)), equity_now)
                gross = open_positions_gross_notional()
                gross_pct = (gross / equity_now) if equity_now > 0 else 0.0
                print(
                    f"[{now_str()}] equity={equity_now:.2f} dd_day={dd:.2%} "
                    f"allow_entries={allow_entries}({reason}) gross={gross_pct:.2%} "
                    f"margin_level={snap.get('margin_level', 0.0):.1f}%"
                )

        except Exception as e:
            print(f"[{now_str()}] ERROR: {e}")

        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    main()
