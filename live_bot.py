import json
import time
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

MARKETS = [
    {"name": "US500", "symbol": "US500.cash"},
    {"name": "US100", "symbol": "US100.cash"},
    {"name": "US30",  "symbol": "US30.cash"},
]

MAX_GROSS_EXPOSURE_TOTAL = 2.0

TF_MARKET_WEIGHTS = {"US500": 0.34, "US100": 0.29, "US30": 0.37}
MR_MARKET_WEIGHTS = {"US500": 1/3,  "US100": 1/3,  "US30": 1/3}

STRATEGY_MARKET_WEIGHTS = {
    "TF": TF_MARKET_WEIGHTS,
    "MR": MR_MARKET_WEIGHTS,
}

STRATEGY_TARGETS_BASE = {
    "TF": 1.0,
    "MR": 1.0,
}

MODE = "EVAL"

EVAL_EXPOSURE_FACTOR = 1.25
FUNDED_EXPOSURE_FACTOR = 1.00

SOFT_CUTOFF_DAILY = -0.04

START_BALANCE_FOR_LIMITS = 50_000
DAILY_LOSS_LIMIT_ABS_FRAC = 0.05
MAX_LOSS_LIMIT_ABS_FRAC = 0.10

TF_PARAMS = dict(exit_confirm_bars=10, adx_threshold=15, ema_fast_len=70, ema_slow_len=120)
MR_PARAMS = dict(ema_fast_len=20, ema_slow_len=250, pullback_frac=0.20)

H1_BARS = 600
D1_BARS = 800

SLEEP_SECONDS = 1.0
HEARTBEAT_EVERY_SEC = 60


# ==========================
# STATE
# ==========================

def load_state():
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except:
        return {}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ==========================
# MT5 DATA
# ==========================

def mt5_timeframe(tf):
    return mt5.TIMEFRAME_H1 if tf == "H1" else mt5.TIMEFRAME_D1


def fetch_ohlc(symbol, tf, n, min_bars=120):

    ensure_symbol(symbol)

    rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe(tf), 0, n)

    if rates is None or len(rates) < min_bars:
        return pd.DataFrame()

    df = pd.DataFrame(rates)

    df["time"] = pd.to_datetime(df["time"], unit="s")

    df = df.set_index("time")

    return df[["open", "high", "low", "close"]]


def last_closed_bar(df):
    return df.iloc[-2]


def prev_closed_bar(df):
    return df.iloc[-3]


# ==========================
# MAGIC
# ==========================

def magic_for(market_name, strategy):

    base = 11000 if strategy == "TF" else 21000

    return base + abs(hash(market_name)) % 1000


# ==========================
# INDICATORS
# ==========================

def compute_tf_indicators(df):

    df = df.copy()

    df["ema_fast"] = df["close"].ewm(span=TF_PARAMS["ema_fast_len"], adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=TF_PARAMS["ema_slow_len"], adjust=False).mean()

    df["adx"] = ta.trend.ADXIndicator(
        df["high"], df["low"], df["close"], window=14
    ).adx()

    return df


def compute_mr_indicators(df):

    df = df.copy()

    df["ema_fast"] = df["close"].ewm(span=MR_PARAMS["ema_fast_len"], adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=MR_PARAMS["ema_slow_len"], adjust=False).mean()

    return df


# ==========================
# SIGNALS
# ==========================

def tf_entry(prev, prev2):

    adx_ok = prev["adx"] > TF_PARAMS["adx_threshold"]

    cross = (prev2["ema_fast"] < prev2["ema_slow"]) and (prev["ema_fast"] > prev["ema_slow"])

    return bool(adx_ok and cross)


def tf_exit(prev):

    return prev["ema_fast"] <= prev["ema_slow"]


def mr_entry(prev):

    close = prev["close"]
    ema_fast = prev["ema_fast"]
    ema_slow = prev["ema_slow"]

    high = prev["high"]
    low = prev["low"]

    regime = (close < ema_fast) and (close > ema_slow)

    deep = close < (low + MR_PARAMS["pullback_frac"] * (high - low))

    return bool(regime and deep)


def mr_exit(prev):

    return prev["high"] >= prev["ema_fast"]


# ==========================
# EXPOSURE
# ==========================

def strategy_targets():

    factor = EVAL_EXPOSURE_FACTOR if MODE == "EVAL" else FUNDED_EXPOSURE_FACTOR

    return {k: v * factor for k, v in STRATEGY_TARGETS_BASE.items()}


def open_positions_gross_notional():

    gross = 0.0

    pos = mt5.positions_get()

    if pos is None:
        return 0.0

    for p in pos:
        gross += abs(p.price_current * p.volume)

    return gross


def remaining_capacity(equity):

    gross = open_positions_gross_notional()

    cap = equity * MAX_GROSS_EXPOSURE_TOTAL

    return max(0.0, cap - gross)


def desired_notional(equity, strat, market):

    targets = strategy_targets()

    w = STRATEGY_MARKET_WEIGHTS[strat][market]

    return equity * targets[strat] * w


# ==========================
# RISK
# ==========================

def day_drawdown(day_start, equity):

    return equity / day_start - 1


def risk_gate(state, equity):

    day_start = state.get("day_start_equity", equity)

    dd = day_drawdown(day_start, equity)

    if dd <= SOFT_CUTOFF_DAILY:
        return False

    daily_abs = DAILY_LOSS_LIMIT_ABS_FRAC * START_BALANCE_FOR_LIMITS

    if equity < day_start - daily_abs:
        return False

    max_floor = START_BALANCE_FOR_LIMITS * (1 - MAX_LOSS_LIMIT_ABS_FRAC)

    if equity < max_floor:
        return False

    return True


# ==========================
# STRATEGY EXECUTION
# ==========================

def run_tf(cfg, state, allow_entries):

    snap = account_snapshot()

    equity = snap["equity"]

    capacity = remaining_capacity(equity)

    for m in MARKETS:

        name = m["name"]
        sym = m["symbol"]

        magic = magic_for(name, "TF")

        pos = get_position(sym, magic)

        df = fetch_ohlc(sym, "H1", H1_BARS)

        if df.empty:
            continue

        df = compute_tf_indicators(df)

        prev = last_closed_bar(df)
        prev2 = prev_closed_bar(df)

        bc_map = state.setdefault("tf_bc", {})

        bc = bc_map.get(name, 0)

        # EXIT
        if pos is not None:

            if tf_exit(prev):
                bc += 1
            else:
                bc = 0

            bc_map[name] = bc

            if bc >= TF_PARAMS["exit_confirm_bars"]:
                close_position_market(sym, magic, cfg, "TF_EXIT")
                bc_map[name] = 0

            continue

        # ENTRY
        if not allow_entries:
            continue

        if capacity <= 0:
            continue

        if tf_entry(prev, prev2):

            notional = desired_notional(equity, "TF", name)

            notional = min(notional, capacity)

            ok, vol = open_long_by_notional(sym, notional, magic, cfg, "TF_ENTRY")

            if ok:
                capacity -= notional


def run_mr(cfg, state, allow_entries):

    snap = account_snapshot()

    equity = snap["equity"]

    capacity = remaining_capacity(equity)

    for m in MARKETS:

        name = m["name"]
        sym = m["symbol"]

        magic = magic_for(name, "MR")

        pos = get_position(sym, magic)

        df = fetch_ohlc(sym, "D1", D1_BARS)

        if df.empty:
            continue

        df = compute_mr_indicators(df)

        prev = last_closed_bar(df)

        # EXIT
        if pos is not None:

            if mr_exit(prev):
                close_position_market(sym, magic, cfg, "MR_EXIT")

            continue

        # ENTRY
        if not allow_entries:
            continue

        if capacity <= 0:
            continue

        if mr_entry(prev):

            notional = desired_notional(equity, "MR", name)

            notional = min(notional, capacity)

            ok, vol = open_long_by_notional(sym, notional, magic, cfg, "MR_ENTRY")

            if ok:
                capacity -= notional


# ==========================
# MAIN LOOP
# ==========================

def main():

    ensure_initialized()

    cfg = ExecConfig()

    state = load_state()

    state.setdefault("day_start_equity", None)
    state.setdefault("day_start_date", None)
    state.setdefault("tf_bc", {})

    print("LIVE BOT STARTED")

    last_heartbeat = 0

    while True:

        try:

            snap = account_snapshot()

            equity = snap["equity"]

            today = date.today().isoformat()

            if state.get("day_start_date") != today:

                state["day_start_date"] = today
                state["day_start_equity"] = equity

                print("New day equity:", equity)

            allow = risk_gate(state, equity)

            run_tf(cfg, state, allow)
            run_mr(cfg, state, allow)

            save_state(state)

            now = time.time()

            if now - last_heartbeat > HEARTBEAT_EVERY_SEC:

                last_heartbeat = now

                dd = day_drawdown(state["day_start_equity"], equity)

                print(
                    f"[{now_str()}] equity={equity:.2f} dd={dd:.2%} allow={allow}"
                )

        except Exception as e:

            print("ERROR:", e)

        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    main()
