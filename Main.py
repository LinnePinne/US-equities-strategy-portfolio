import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
from dataclasses import dataclass

plt.style.use("default")

# ==========================
# KONFIG: MARKNADER & FILER
# ==========================
# Trend = 1H-data (master för MTM)
# MR    = Daily-data (signal), men trades alignas till 1H fills för portfölj/FTMO-MTM
markets = [
    {"name": "US500", "csv_1h": "US500_1H_2012-now.csv", "csv_1d": "US500_1D_2012-2025.csv"},
    {"name": "US100", "csv_1h": "USTEC_1H_2012-now.csv", "csv_1d": "USTEC_1D_2012-2025.csv"},
    {"name": "US30",  "csv_1h": "US30_1H_2012-now.csv",  "csv_1d": "US30_1D_2012-2025.csv"},
]

TF_MARKET_WEIGHTS = {"US500": 0.34, "US100": 0.29, "US30": 0.37}
MR_MARKET_WEIGHTS = {"US500": 1/3,  "US100": 1/3,  "US30": 1/3}

STRATEGY_MARKET_WEIGHTS = {
    "TF": TF_MARKET_WEIGHTS,
    "MR": MR_MARKET_WEIGHTS,
}

# ==========================
# PORTFÖLJ / EXPOSURE
# ==========================
START_CAPITAL = 50_000

# Total cap för portföljens gross exposure (t.ex. 2.0 = 200% av equity).
MAX_GROSS_EXPOSURE_TOTAL = 2.0

# Per strategi target exposure (budgetar inom total cap):
# Trend: 1.25 (enligt er vol-target)
# MR: välj konservativt initialt när den kombineras med trend (rekommenderat 0.50–0.75).
STRATEGY_TARGETS = {
    "TF": 1.0,
    "MR": 1.0,
}

# ===== FTMO / Pipeline settings =====
EVAL_EXPOSURE_FACTOR = 1.5   # t.ex. 1.15–1.40, ni justerar
FUNDED_EXPOSURE_FACTOR = 1.00 # alltid 1.0 i funded enligt er plan

SOFT_CUTOFF_DAILY = -0.04     # -4% cutoff (stoppa losses resten av dagen)

CHALLENGE_TARGET = 0.10
VERIFICATION_TARGET = 0.05
DAILY_LOSS_LIMIT = 0.05       # -5%
MAX_LOSS_LIMIT = 0.10         # -10%
MIN_TRADING_DAYS_EVAL = 4
PAYOUT_WAIT_DAYS = 14

BOOT_N_ITER = 20000
BOOT_BLOCK_LEN = 20
BOOT_HORIZON_DAYS = 365 * 3
BOOT_SEED = 42

# ==========================
# INSTRUMENT: $ per indexpunkt per kontrakt
# ==========================
POINT_VALUE = {"US500": 1.0, "US100": 1.0, "US30": 1.0}

# ==========================
# COST MODEL (POINTS)
# ==========================
HALF = 0.5
SLIPPAGE_POINTS = 0.5
FIXED_SPREAD_POINTS = 0.8
COMM_POINTS_PER_SIDE = 0.05  # per side

def commission_round_turn_points():
    return 2.0 * COMM_POINTS_PER_SIDE

# ============================================================
# DATA LOADING
# ============================================================

def load_market_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")
    else:
        raise ValueError("Hittar ingen 'timestamp' eller 'datetime' i CSV.")

    df = df.sort_index()

    required_cols = {"open", "high", "low", "close"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV måste innehålla: {required_cols}")

    return df

# ============================================================
# TREND STRATEGY (1H) - er funktion (oförändrad logik)
# ============================================================

def generate_trades_for_market_trend(
    market_name: str,
    df: pd.DataFrame,
    exit_confirm_bars: int = 10,
    adx_threshold: float = 15,
    ema_fast_len: int = 70,
    ema_slow_len: int = 120,
) -> pd.DataFrame:

    df = df.copy()

    df["ema_fast"] = df["close"].ewm(span=ema_fast_len, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=ema_slow_len, adjust=False).mean()

    adx_len = 14
    df["adx"] = ta.trend.ADXIndicator(
        high=df["high"], low=df["low"], close=df["close"],
        window=adx_len, fillna=False
    ).adx()

    use_spread_col = "spread_points" in df.columns

    def spread_points(row) -> float:
        return float(row["spread_points"]) if use_spread_col else FIXED_SPREAD_POINTS

    trades = []

    in_position = False
    entry_price = None
    entry_signal_time = None
    entry_fill_time = None
    exit_breach_count = 0

    idx = df.index.to_list()

    for i in range(1, len(df) - 1):
        ts = idx[i]
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]
        next_row = df.iloc[i + 1]

        if np.isnan(row["adx"]) or np.isnan(row["ema_slow"]) or np.isnan(prev_row["ema_slow"]):
            continue

        # Exit
        if in_position:
            if row["ema_fast"] <= row["ema_slow"]:
                exit_breach_count += 1
            else:
                exit_breach_count = 0

            if exit_breach_count >= exit_confirm_bars:
                spr = spread_points(next_row)
                exit_fill_price = float(next_row["open"] - HALF * spr - SLIPPAGE_POINTS)
                exit_fill_time = idx[i + 1]

                trades.append({
                    "Market": market_name,
                    "Strategy": "TF",
                    "Direction": "LONG",
                    "Entry Signal Time": entry_signal_time,
                    "Entry Fill Time": entry_fill_time,
                    "Exit Fill Time": exit_fill_time,
                    "Entry Price": float(entry_price),
                    "Exit Price": float(exit_fill_price),
                    "Exit Reason": f"ema_recross_{exit_confirm_bars}bars",
                    "Comm RT (points)": float(commission_round_turn_points()),
                })

                in_position = False
                entry_price = None
                entry_signal_time = None
                entry_fill_time = None
                exit_breach_count = 0

            if in_position:
                continue

        # Entry
        adx_ok = row["adx"] > adx_threshold
        cross_up = (prev_row["ema_fast"] < prev_row["ema_slow"]) and (row["ema_fast"] > row["ema_slow"])

        if cross_up and adx_ok:
            spr = spread_points(next_row)
            entry_fill_price = float(next_row["open"] + HALF * spr + SLIPPAGE_POINTS)

            in_position = True
            entry_signal_time = ts
            entry_fill_time = idx[i + 1]
            entry_price = entry_fill_price
            exit_breach_count = 0

    # Forced exit at end
    if in_position:
        last_row = df.iloc[-1]
        spr = float(last_row["spread_points"]) if "spread_points" in df.columns else FIXED_SPREAD_POINTS
        exit_fill_price = float(last_row["close"] - HALF * spr - SLIPPAGE_POINTS)
        exit_fill_time = df.index[-1]

        trades.append({
            "Market": market_name,
            "Strategy": "TF",
            "Direction": "LONG",
            "Entry Signal Time": entry_signal_time,
            "Entry Fill Time": entry_fill_time,
            "Exit Fill Time": exit_fill_time,
            "Entry Price": float(entry_price),
            "Exit Price": float(exit_fill_price),
            "Exit Reason": "forced_exit_end_of_test",
            "Comm RT (points)": float(commission_round_turn_points()),
        })

    return pd.DataFrame(trades)

# ============================================================
# MEAN REVERSION STRATEGY (DAILY SIGNAL)
# ============================================================

def generate_trades_for_market_mr_daily(
    market_name: str,
    df: pd.DataFrame,
    ema_fast_len: int = 20,
    ema_slow_len: int = 250,
    pullback_frac: float = 0.20,  # deep pullback: close < low + frac*(high-low)
) -> pd.DataFrame:
    """
    MR (LONG only) på daily:
      Regim: close < ema_fast och close > ema_slow
      Trigger: deep pullback inom dagens range
      Entry fill: nästa dags open (i+1) + cost (ask)  [på daily-nivå]
      Exit trigger: dagens high >= ema_fast => exit fill nästa dags open (bid)
    OBS: Dessa trades alignas senare till 1H för portfölj/MTM.
    """
    df = df.copy()
    df["ema_fast"] = df["close"].ewm(span=ema_fast_len, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=ema_slow_len, adjust=False).mean()

    use_spread_col = "spread_points" in df.columns

    def spread_points(row) -> float:
        return float(row["spread_points"]) if use_spread_col else FIXED_SPREAD_POINTS

    trades = []

    in_position = False
    entry_signal_time = None
    entry_fill_time = None
    entry_price = None

    idx = df.index.to_list()

    for i in range(1, len(df) - 1):
        ts = idx[i]
        row = df.iloc[i]
        next_row = df.iloc[i + 1]

        if np.isnan(row["ema_slow"]) or np.isnan(row["ema_fast"]):
            continue

        # Exit
        if in_position:
            if float(row["high"]) >= float(row["ema_fast"]):
                spr = spread_points(next_row)
                exit_fill_price = float(next_row["open"] - HALF * spr - SLIPPAGE_POINTS)
                exit_fill_time = idx[i + 1]

                trades.append({
                    "Market": market_name,
                    "Strategy": "MR",
                    "Direction": "LONG",
                    "Entry Signal Time": entry_signal_time,
                    "Entry Fill Time": entry_fill_time,
                    "Exit Fill Time": exit_fill_time,
                    "Entry Price": float(entry_price),
                    "Exit Price": float(exit_fill_price),
                    "Exit Reason": "ema_fast_touch",
                    "Comm RT (points)": float(commission_round_turn_points()),
                })

                in_position = False
                entry_signal_time = None
                entry_fill_time = None
                entry_price = None

            if in_position:
                continue

        # Entry
        close_px = float(row["close"])
        ema_fast = float(row["ema_fast"])
        ema_slow = float(row["ema_slow"])
        high = float(row["high"])
        low = float(row["low"])

        bullish_pullback_regime = (close_px < ema_fast) and (close_px > ema_slow)
        deep_pullback = close_px < (low + pullback_frac * (high - low))

        if bullish_pullback_regime and deep_pullback:
            spr = spread_points(next_row)
            entry_fill_price = float(next_row["open"] + HALF * spr + SLIPPAGE_POINTS)

            in_position = True
            entry_signal_time = ts
            entry_fill_time = idx[i + 1]
            entry_price = entry_fill_price

    # Forced exit at end
    if in_position:
        last_row = df.iloc[-1]
        spr = float(last_row["spread_points"]) if "spread_points" in df.columns else FIXED_SPREAD_POINTS
        exit_fill_price = float(last_row["close"] - HALF * spr - SLIPPAGE_POINTS)
        exit_fill_time = df.index[-1]

        trades.append({
            "Market": market_name,
            "Strategy": "MR",
            "Direction": "LONG",
            "Entry Signal Time": entry_signal_time,
            "Entry Fill Time": entry_fill_time,
            "Exit Fill Time": exit_fill_time,
            "Entry Price": float(entry_price),
            "Exit Price": float(exit_fill_price),
            "Exit Reason": "forced_exit_end_of_test",
            "Comm RT (points)": float(commission_round_turn_points()),
        })

    return pd.DataFrame(trades)

# ============================================================
# ALIGN DAILY TRADES TO 1H FILLS (för MTM på 1H)
# ============================================================

def align_trades_to_hourly_opens(
    trades_df: pd.DataFrame,
    hourly_df: pd.DataFrame,
    half: float = HALF,
    slippage_points: float = SLIPPAGE_POINTS,
    fixed_spread_points: float = FIXED_SPREAD_POINTS,
) -> pd.DataFrame:
    """
    Alignar Entry/Exit Fill Times till nästa tillgängliga 1H timestamp och
    re-prisar fills med 1H open +/- spread/slippage.
    """
    if trades_df.empty:
        return trades_df.copy()

    t = trades_df.copy()
    t["Entry Fill Time"] = pd.to_datetime(t["Entry Fill Time"])
    t["Exit Fill Time"] = pd.to_datetime(t["Exit Fill Time"])

    hdf = hourly_df.sort_index()
    hidx = hdf.index
    if not isinstance(hidx, pd.DatetimeIndex):
        hidx = pd.to_datetime(hidx)

    use_spread_col = "spread_points" in hdf.columns

    def next_ts(ts):
        pos = hidx.searchsorted(ts, side="left")
        if pos >= len(hidx):
            return hidx[-1]
        return hidx[pos]

    def spread_at(ts):
        if use_spread_col:
            return float(hdf.loc[ts, "spread_points"])
        return float(fixed_spread_points)

    new_entry_ts, new_exit_ts, new_entry_px, new_exit_px = [], [], [], []

    for _, r in t.iterrows():
        direction = str(r.get("Direction", "LONG")).upper()

        et = next_ts(r["Entry Fill Time"])
        xt = next_ts(r["Exit Fill Time"])

        eopen = float(hdf.loc[et, "open"])
        xopen = float(hdf.loc[xt, "open"])

        spr_e = spread_at(et)
        spr_x = spread_at(xt)

        if direction == "LONG":
            efill = eopen + half * spr_e + slippage_points
            xfill = xopen - half * spr_x - slippage_points
        else:
            efill = eopen - half * spr_e - slippage_points
            xfill = xopen + half * spr_x + slippage_points

        new_entry_ts.append(et)
        new_exit_ts.append(xt)
        new_entry_px.append(efill)
        new_exit_px.append(xfill)

    t["Entry Fill Time"] = new_entry_ts
    t["Exit Fill Time"] = new_exit_ts
    t["Entry Price"] = new_entry_px
    t["Exit Price"] = new_exit_px

    return t

# ============================================================
# MULTI-STRATEGY PORTFOLIO MTM (1H master)
# ============================================================

def build_portfolio_mtm_cash_multi_strategy(
    market_dfs_1h: dict,
    trades_df: pd.DataFrame,
    start_capital: float,
    max_gross_exposure: float,
    strategy_targets: dict,
    strategy_market_weights: dict | None = None,  # <-- NY
    allow_leverage: bool = True,
) -> tuple:

    tr = trades_df.copy()
    tr["Entry Fill Time"] = pd.to_datetime(tr["Entry Fill Time"])
    tr["Exit Fill Time"] = pd.to_datetime(tr["Exit Fill Time"])

    required = {"Market", "Strategy", "Direction", "Entry Fill Time", "Exit Fill Time", "Entry Price", "Exit Price"}
    missing = required - set(tr.columns)
    if missing:
        raise ValueError(f"Trades saknar kolumner: {missing}")

    # Master index från 1H
    all_index = None
    for mkt, df in market_dfs_1h.items():
        dfx = df.sort_index()
        if dfx.index.has_duplicates:
            dfx = dfx[~dfx.index.duplicated(keep="last")]
        all_index = dfx.index if all_index is None else all_index.union(dfx.index)
    all_index = pd.DatetimeIndex(all_index.sort_values().unique())

    # Close matrix
    closes = pd.DataFrame(index=all_index)
    for mkt, df in market_dfs_1h.items():
        dfx = df.sort_index()
        if dfx.index.has_duplicates:
            dfx = dfx[~dfx.index.duplicated(keep="last")]
        closes[mkt] = dfx["close"].reindex(all_index).ffill()

    mkts = sorted(market_dfs_1h.keys())

    def normalize_weights(w: dict) -> dict:
        s = sum(w.get(m, 0.0) for m in mkts)
        if s <= 0:
            raise ValueError("Vikter summerar till 0.")
        return {m: float(w.get(m, 0.0)) / s for m in mkts}

    # Default: equal weights för alla strategier som förekommer
    if strategy_market_weights is None:
        eq = {m: 1.0 / len(mkts) for m in mkts}
        strategy_market_weights = {"TF": eq, "MR": eq}
    else:
        # säkerställ att alla marknader finns och normalisera per strategi
        for strat, w in strategy_market_weights.items():
            missing = set(mkts) - set(w.keys())
            if missing:
                raise ValueError(f"strategy_market_weights[{strat}] saknar marknader: {missing}")
            strategy_market_weights[strat] = normalize_weights(w)

    print("\n[Portfolio] Strategy market weights:")
    for strat in sorted(strategy_market_weights.keys()):
        print(f"  {strat}:")
        for m in mkts:
            print(f"    {m}: {strategy_market_weights[strat][m]:.4f}")

    print("\n[Portfolio] Strategy targets (gross exposure):")
    for k in sorted(strategy_targets.keys()):
        print(f"  {k}: {float(strategy_targets[k]):.4f}")

    entries = tr.sort_values(["Entry Fill Time", "Market", "Strategy"]).groupby("Entry Fill Time")
    exits = tr.sort_values(["Exit Fill Time", "Market", "Strategy"]).groupby("Exit Fill Time")

    cash = float(start_capital)
    realized_equity = float(start_capital)

    # positions keyed by (market, strategy)
    positions = {}  # (mkt, strat) -> dict(contracts, entry_price, direction)

    equity_path = []
    realized_path = []
    gross_exposure_path = []
    open_positions_path = []

    def mtm_and_gross(ts):
        mtm = 0.0
        gross = 0.0
        for (mkt, strat), pos in positions.items():
            px = float(closes.loc[ts, mkt])
            pv = float(POINT_VALUE[mkt])
            direction = pos["direction"]
            sign = 1.0 if direction == "LONG" else -1.0
            notional = float(pos["contracts"]) * px * pv
            gross += abs(notional)
            mtm += sign * notional
        return mtm, gross

    for ts in all_index:
        # 1) Exits
        if ts in exits.groups:
            block = exits.get_group(ts)
            for _, t in block.iterrows():
                mkt = t["Market"]
                strat = t["Strategy"]
                key = (mkt, strat)
                if key not in positions:
                    continue

                pos = positions[key]
                contracts = float(pos["contracts"])
                entry_price = float(pos["entry_price"])
                exit_price = float(t["Exit Price"])
                pv = float(POINT_VALUE[mkt])

                direction = pos["direction"]
                sign = 1.0 if direction == "LONG" else -1.0

                pnl_gross = sign * (exit_price - entry_price) * contracts * pv
                comm_points = float(t.get("Comm RT (points)", 0.0))
                pnl_comm = comm_points * pv * contracts
                pnl_net = pnl_gross - pnl_comm

                cash += sign * (contracts * exit_price * pv)
                cash -= pnl_comm
                realized_equity += pnl_net

                del positions[key]

        # 2) Entries
        if ts in entries.groups:
            block = entries.get_group(ts)

            for _, t in block.iterrows():
                mkt = t["Market"]
                strat = t["Strategy"]
                direction = str(t["Direction"]).upper()
                key = (mkt, strat)

                if key in positions:
                    continue

                strat_target = float(strategy_targets.get(strat, 0.0))
                if strat_target <= 0:
                    continue

                entry_price = float(t["Entry Price"])
                pv = float(POINT_VALUE[mkt])

                mtm_value, gross_value = mtm_and_gross(ts)
                equity_now = cash + mtm_value
                remaining_notional_total = max(0.0, equity_now * max_gross_exposure - gross_value)
                if remaining_notional_total <= 0:
                    continue

                w_mkt = float(strategy_market_weights.get(strat, {}).get(mkt, 0.0))
                desired_notional = equity_now * strat_target * w_mkt
                position_notional = min(desired_notional, remaining_notional_total)

                if not allow_leverage:
                    position_notional = min(position_notional, cash)

                denom = entry_price * pv
                contracts = (position_notional / denom) if denom > 0 else 0.0
                if contracts <= 0:
                    continue

                sign = 1.0 if direction == "LONG" else -1.0
                cash -= sign * (contracts * entry_price * pv)

                positions[key] = {"contracts": contracts, "entry_price": entry_price, "direction": direction}

        # 3) MTM snapshot
        mtm_value, gross_value = mtm_and_gross(ts)
        equity = cash + mtm_value

        equity_path.append((ts, equity))
        realized_path.append((ts, realized_equity))
        open_positions_path.append((ts, len(positions)))
        gross_exposure_path.append((ts, gross_value / equity if equity > 0 else 0.0))

    equity_series = pd.Series(
        [v for _, v in equity_path],
        index=pd.DatetimeIndex([t for t, _ in equity_path]),
        name="PortfolioEquity"
    )
    realized_series = pd.Series(
        [v for _, v in realized_path],
        index=pd.DatetimeIndex([t for t, _ in realized_path]),
        name="RealizedEquity"
    )
    open_pos_series = pd.Series(
        [v for _, v in open_positions_path],
        index=pd.DatetimeIndex([t for t, _ in open_positions_path]),
        name="OpenPositions"
    )
    gross_exposure_series = pd.Series(
        [v for _, v in gross_exposure_path],
        index=pd.DatetimeIndex([t for t, _ in gross_exposure_path]),
        name="GrossExposurePct"
    )

    daily_equity = equity_series.resample("1D").last().dropna()
    daily_returns = daily_equity.pct_change().dropna()

    return equity_series, realized_series, daily_returns, open_pos_series, gross_exposure_series

# ============================================================
# METRICS / DD HELPERS
# ============================================================

def portfolio_metrics_from_equity(equity_series: pd.Series, daily_returns: pd.Series, trading_days=252) -> dict:
    n_days = (equity_series.index[-1] - equity_series.index[0]).days
    years = n_days / 365.25 if n_days > 0 else np.nan
    cagr = (equity_series.iloc[-1] / equity_series.iloc[0]) ** (1.0 / years) - 1.0 if years and years > 0 else np.nan

    roll_max = equity_series.cummax()
    dd = equity_series / roll_max - 1.0
    max_dd = dd.min()

    mu = daily_returns.mean()
    sd = daily_returns.std(ddof=1)
    sharpe = (mu / sd) * np.sqrt(trading_days) if sd and sd > 0 else np.nan

    calmar = (cagr / abs(max_dd)) if pd.notna(cagr) and pd.notna(max_dd) and max_dd < 0 else np.nan

    return {
        "Equity Start": float(equity_series.iloc[0]),
        "Equity End": float(equity_series.iloc[-1]),
        "CAGR": float(cagr) if pd.notna(cagr) else np.nan,
        "Max Drawdown %": float(max_dd * 100.0) if pd.notna(max_dd) else np.nan,
        "Sharpe (ann.)": float(sharpe) if pd.notna(sharpe) else np.nan,
        "Calmar": float(calmar) if pd.notna(calmar) else np.nan,
        "Avg Daily Return": float(mu) if pd.notna(mu) else np.nan,
        "Daily Vol": float(sd) if pd.notna(sd) else np.nan,
    }

def max_drawdown_pct(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return float(dd.min())

# ============================================================
# FTMO DAILY LOSS / INTRADAY DRAWDOWN ANALYSIS
# ============================================================

def compute_intraday_drawdowns(equity_series: pd.Series) -> pd.DataFrame:
    eq = equity_series.copy().dropna()
    eq.index = pd.to_datetime(eq.index)

    rows = []
    for day, s in eq.groupby(eq.index.date):
        if len(s) < 2:
            continue

        day_start = float(s.iloc[0])
        running_max = s.cummax()
        intraday_dd = (s - running_max) / day_start

        rows.append({
            "date": pd.Timestamp(day),
            "DayStartEquity": day_start,
            "MinIntradayDD": float(intraday_dd.min()),
        })

    return pd.DataFrame(rows).set_index("date")


def daily_loss_statistics(
    intraday_dd_df: pd.DataFrame,
    soft_limit: float = -0.05,
    hard_limit: float = -0.10,
) -> dict:

    n = len(intraday_dd_df)
    if n == 0:
        raise ValueError("Ingen intraday drawdown-data.")

    breach_5 = intraday_dd_df["MinIntradayDD"] <= soft_limit
    breach_10 = intraday_dd_df["MinIntradayDD"] <= hard_limit

    return {
        "Days": n,
        "WorstDayDD": float(intraday_dd_df["MinIntradayDD"].min()),
        "AvgWorstDD": float(intraday_dd_df["MinIntradayDD"].mean()),
        "P(Breach -5%)": float(breach_5.mean()),
        "P(Breach -10%)": float(breach_10.mean()),
        "Count Breach -5%": int(breach_5.sum()),
        "Count Breach -10%": int(breach_10.sum()),
    }



@dataclass
class FTMOParams2:
    challenge_target: float = 0.10
    verification_target: float = 0.05
    daily_loss_limit: float = 0.05
    max_loss_limit: float = 0.10
    min_trading_days_eval: int = 4
    payout_wait_days: int = 14
    soft_cutoff: float = -0.04              # -4% cutoff (intraday)
    eval_exposure_factor: float = 1.5      # >1 för snabbare evaluation
    funded_exposure_factor: float = 1.0     # 1.0 i funded


def build_day_path_library(
    equity_1h: pd.Series,
    trades_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Skapar en dags-DataFrame där varje dag innehåller en intraday path
    som fraktion relativt dagens start:
        frac_path[t] = equity_t / equity_day_start - 1
    Samt flagga om dagen är trading day (minst 1 entry den dagen).
    """
    eq = equity_1h.dropna().copy()
    eq.index = pd.to_datetime(eq.index)
    eq = eq.sort_index()

    # Trading day flag via entries
    tr = trades_df.copy()
    tr["Entry Fill Time"] = pd.to_datetime(tr["Entry Fill Time"])
    entry_dates = set(pd.Timestamp(d) for d in tr["Entry Fill Time"].dt.normalize().unique())

    rows = []
    for day, s in eq.groupby(eq.index.normalize()):
        s = s.sort_index()
        if len(s) < 2:
            continue

        day_start = float(s.iloc[0])
        frac_path = (s / day_start - 1.0).values.astype(float)  # intraday fractional path
        rows.append({
            "date": day,
            "IsTradingDay": bool(day in entry_dates),
            "FracPath": frac_path,
        })

    lib = pd.DataFrame(rows).set_index("date").sort_index()
    return lib


def block_bootstrap_day_indices(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    out = []
    while len(out) < n:
        start = int(rng.integers(0, n))
        block = [(start + j) % n for j in range(block_len)]
        out.extend(block)
    return np.array(out[:n], dtype=int)


def apply_exposure_and_cutoff(
    frac_path: np.ndarray,
    exposure_factor: float,
    soft_cutoff: float,
) -> tuple[np.ndarray, float]:
    """
    Tar en frac_path (relativt dagens start) och skalar den med exposure_factor.
    Om equity når soft_cutoff (t.ex -0.04), plattas equity resten av dagen till cutoff-nivån.
    Returnerar:
      - adjusted frac_path
      - worst intraday dd (min frac)
    """
    p = frac_path.astype(float) * float(exposure_factor)

    if soft_cutoff is not None:
        # cutoff triggers when frac <= soft_cutoff
        hit = np.where(p <= soft_cutoff)[0]
        if hit.size > 0:
            j = int(hit[0])
            p[j:] = soft_cutoff  # flat resten av dagen

    worst = float(np.min(p))
    return p, worst


def simulate_stage_from_daypaths(
    day_sample: pd.DataFrame,      # columns: IsTradingDay, FracPath
    start_balance: float,
    target_frac: float,
    daily_loss_limit_frac: float,
    max_loss_limit_frac: float,
    min_trading_days: int,
    exposure_factor: float,
    soft_cutoff: float,
) -> dict:
    """
    Simulerar Challenge/Verification på dags-path nivå.
    Daily loss check: min equity under dagen jämfört med day's start equity
                      men limit definieras som % av initial balance (FTMO-stil).
    """
    bal0 = float(start_balance)
    equity = bal0

    trading_days = 0
    days_elapsed = 0

    target_equity = bal0 * (1.0 + target_frac)
    daily_loss_abs = bal0 * daily_loss_limit_frac
    max_loss_floor = bal0 * (1.0 - max_loss_limit_frac)

    for dt, row in day_sample.iterrows():
        days_elapsed += 1

        frac_path = row["FracPath"]
        adj_path, worst_frac = apply_exposure_and_cutoff(frac_path, exposure_factor, soft_cutoff)

        # Daily loss check (in $): worst intraday drop relative to day's start equity (equity at day start)
        day_start_equity = equity
        min_equity_intraday = day_start_equity * (1.0 + worst_frac)
        intraday_drop_abs = day_start_equity - min_equity_intraday

        if intraday_drop_abs > daily_loss_abs:
            return {
                "passed": False, "failed": True, "fail_reason": "MaxDailyLoss",
                "days_elapsed": days_elapsed, "trading_days": trading_days,
                "end_equity": float(min_equity_intraday), "pass_day": None
            }

        # End-of-day equity: last value of adjusted path
        end_frac = float(adj_path[-1])
        equity = day_start_equity * (1.0 + end_frac)

        if bool(row["IsTradingDay"]):
            trading_days += 1

        # Max loss (overall)
        if equity < max_loss_floor:
            return {
                "passed": False, "failed": True, "fail_reason": "MaxLoss",
                "days_elapsed": days_elapsed, "trading_days": trading_days,
                "end_equity": float(equity), "pass_day": None
            }

        # Pass condition
        if equity >= target_equity and trading_days >= min_trading_days:
            return {
                "passed": True, "failed": False, "fail_reason": None,
                "days_elapsed": days_elapsed, "trading_days": trading_days,
                "end_equity": float(equity), "pass_day": days_elapsed
            }

    return {
        "passed": False, "failed": False, "fail_reason": "HorizonEnded",
        "days_elapsed": days_elapsed, "trading_days": trading_days,
        "end_equity": float(equity), "pass_day": None
    }


def simulate_funded_to_payout_from_daypaths(
    day_sample: pd.DataFrame,
    start_balance: float,
    daily_loss_limit_frac: float,
    max_loss_limit_frac: float,
    payout_wait_days: int,
    exposure_factor: float,
    soft_cutoff: float,
) -> dict:
    """
    Simulerar Funded tills payout eligibility:
      - earliest payout_wait_days efter första trading day
      - equity > start_balance
      - inga riskbrott
    """
    bal0 = float(start_balance)
    equity = bal0

    daily_loss_abs = bal0 * daily_loss_limit_frac
    max_loss_floor = bal0 * (1.0 - max_loss_limit_frac)

    days_elapsed = 0
    first_trade_day_idx = None

    for dt, row in day_sample.iterrows():
        days_elapsed += 1

        if first_trade_day_idx is None and bool(row["IsTradingDay"]):
            first_trade_day_idx = days_elapsed

        frac_path = row["FracPath"]
        adj_path, worst_frac = apply_exposure_and_cutoff(frac_path, exposure_factor, soft_cutoff)

        day_start_equity = equity
        min_equity_intraday = day_start_equity * (1.0 + worst_frac)
        intraday_drop_abs = day_start_equity - min_equity_intraday

        if intraday_drop_abs > daily_loss_abs:
            return {"eligible": False, "fail_reason": "MaxDailyLoss", "days_elapsed": days_elapsed, "eligible_day": None}

        equity = day_start_equity * (1.0 + float(adj_path[-1]))

        if equity < max_loss_floor:
            return {"eligible": False, "fail_reason": "MaxLoss", "days_elapsed": days_elapsed, "eligible_day": None}

        if first_trade_day_idx is not None:
            if (days_elapsed - first_trade_day_idx) >= payout_wait_days and equity > bal0:
                return {"eligible": True, "fail_reason": None, "days_elapsed": days_elapsed, "eligible_day": days_elapsed}

    return {"eligible": False, "fail_reason": "HorizonEnded", "days_elapsed": days_elapsed, "eligible_day": None}


def run_bootstrap_ftmo_pipeline_daypaths(
    day_lib: pd.DataFrame,
    start_balance: float,
    n_iter: int,
    horizon_days: int,
    block_len: int,
    seed: int,
    params: FTMOParams2,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = day_lib.copy()
    n = len(base)
    if n < 2 * block_len:
        raise ValueError("För få dagar i day_lib relativt block_len.")

    results = []

    L = min(horizon_days, n)

    for k in range(n_iter):
        idx = block_bootstrap_day_indices(n=L, block_len=block_len, rng=rng)
        sample = base.iloc[idx].copy()

        # Challenge (högre exposure)
        ch = simulate_stage_from_daypaths(
            day_sample=sample,
            start_balance=start_balance,
            target_frac=params.challenge_target,
            daily_loss_limit_frac=params.daily_loss_limit,
            max_loss_limit_frac=params.max_loss_limit,
            min_trading_days=params.min_trading_days_eval,
            exposure_factor=params.eval_exposure_factor,
            soft_cutoff=params.soft_cutoff,
        )
        if not ch["passed"]:
            results.append({
                "iter": k,
                "passed_challenge": False,
                "passed_verification": False,
                "reached_payout": False,
                "days_to_challenge": ch["days_elapsed"],
                "days_to_verification": np.nan,
                "days_to_payout": np.nan,
                "fail_stage": "Challenge",
                "fail_reason": ch["fail_reason"],
            })
            continue

        # Remaining path after challenge
        rem = sample.iloc[ch["days_elapsed"]:]
        if len(rem) < 50:
            idx2 = block_bootstrap_day_indices(n=L, block_len=block_len, rng=rng)
            rem = base.iloc[idx2].copy()

        # Verification (högre exposure)
        ver = simulate_stage_from_daypaths(
            day_sample=rem,
            start_balance=start_balance,
            target_frac=params.verification_target,
            daily_loss_limit_frac=params.daily_loss_limit,
            max_loss_limit_frac=params.max_loss_limit,
            min_trading_days=params.min_trading_days_eval,
            exposure_factor=params.eval_exposure_factor,
            soft_cutoff=params.soft_cutoff,
        )
        if not ver["passed"]:
            results.append({
                "iter": k,
                "passed_challenge": True,
                "passed_verification": False,
                "reached_payout": False,
                "days_to_challenge": ch["days_elapsed"],
                "days_to_verification": ver["days_elapsed"],
                "days_to_payout": np.nan,
                "fail_stage": "Verification",
                "fail_reason": ver["fail_reason"],
            })
            continue

        rem2 = rem.iloc[ver["days_elapsed"]:]
        if len(rem2) < 50:
            idx3 = block_bootstrap_day_indices(n=L, block_len=block_len, rng=rng)
            rem2 = base.iloc[idx3].copy()

        # Funded (tillbaka till 1.0 exposure)
        fund = simulate_funded_to_payout_from_daypaths(
            day_sample=rem2,
            start_balance=start_balance,
            daily_loss_limit_frac=params.daily_loss_limit,
            max_loss_limit_frac=params.max_loss_limit,
            payout_wait_days=params.payout_wait_days,
            exposure_factor=params.funded_exposure_factor,
            soft_cutoff=params.soft_cutoff,  # ni kan sätta None om ni INTE vill cutoff i funded
        )

        results.append({
            "iter": k,
            "passed_challenge": True,
            "passed_verification": True,
            "reached_payout": bool(fund["eligible"]),
            "days_to_challenge": ch["days_elapsed"],
            "days_to_verification": ver["days_elapsed"],
            "days_to_payout": fund["eligible_day"] if fund["eligible"] else np.nan,
            "fail_stage": "Funded" if not fund["eligible"] else None,
            "fail_reason": fund["fail_reason"] if not fund["eligible"] else None,
        })

    return pd.DataFrame(results)


def summarize_times(df: pd.DataFrame, col: str) -> dict:
    s = df[col].dropna()
    if s.empty:
        return {"count": 0}
    return {
        "count": int(s.shape[0]),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "p05": float(s.quantile(0.05)),
        "p95": float(s.quantile(0.95)),
    }

# ============================================================
# MAIN
# ============================================================

def main():
    # 1) Load data
    market_dfs_1h = {}
    market_dfs_1d = {}

    for m in markets:
        name = m["name"]
        market_dfs_1h[name] = load_market_df(m["csv_1h"])
        market_dfs_1d[name] = load_market_df(m["csv_1d"])

    mkts = sorted(market_dfs_1h.keys())

    # 2) Generate trades
    tf_trades_list = []
    mr_trades_daily_list = []

    for mkt in mkts:
        df1h = market_dfs_1h[mkt]
        df1d = market_dfs_1d[mkt]

        # Trend trades (1H)
        t_tf = generate_trades_for_market_trend(
            market_name=mkt,
            df=df1h,
            exit_confirm_bars=10,
            adx_threshold=15,
            ema_fast_len=70,
            ema_slow_len=120,
        )
        if not t_tf.empty:
            tf_trades_list.append(t_tf)

        # MR trades (Daily signal)
        t_mr = generate_trades_for_market_mr_daily(
            market_name=mkt,
            df=df1d,
            ema_fast_len=20,
            ema_slow_len=250,
            pullback_frac=0.20,
        )
        if not t_mr.empty:
            mr_trades_daily_list.append(t_mr)

    tf_trades = pd.concat(tf_trades_list, ignore_index=True) if tf_trades_list else pd.DataFrame()
    mr_trades_daily = pd.concat(mr_trades_daily_list, ignore_index=True) if mr_trades_daily_list else pd.DataFrame()

    if tf_trades.empty and mr_trades_daily.empty:
        raise RuntimeError("Inga trades genererades för någon strategi.")

    # 3) Align MR trades to 1H fills (per market)
    mr_trades_aligned_list = []
    if not mr_trades_daily.empty:
        for mkt in mkts:
            tdf = mr_trades_daily[mr_trades_daily["Market"] == mkt].copy()
            if tdf.empty:
                continue
            aligned = align_trades_to_hourly_opens(tdf, market_dfs_1h[mkt])
            mr_trades_aligned_list.append(aligned)

    mr_trades_1h = pd.concat(mr_trades_aligned_list, ignore_index=True) if mr_trades_aligned_list else pd.DataFrame()

    # 4) Combine trades
    portfolio_trades = pd.concat([df for df in [tf_trades, mr_trades_1h] if not df.empty], ignore_index=True)

    # 5) Portfolio MTM simulation (1H master)
    equity_series, realized_equity_series, daily_returns, open_pos_series, gross_exposure_series = build_portfolio_mtm_cash_multi_strategy(
        market_dfs_1h=market_dfs_1h,
        trades_df=portfolio_trades,
        start_capital=START_CAPITAL,
        max_gross_exposure=MAX_GROSS_EXPOSURE_TOTAL,
        strategy_targets=STRATEGY_TARGETS,
        strategy_market_weights=STRATEGY_MARKET_WEIGHTS,  # <-- här
        allow_leverage=True,
    )

    # 6) Metrics
    metrics = portfolio_metrics_from_equity(equity_series, daily_returns)
    mtm_dd = max_drawdown_pct(equity_series)
    closed_dd = max_drawdown_pct(realized_equity_series)

    print("\n--- COMBINED PORTFÖLJ METRICS (MTM, 1H master) ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    print("\nSanity: Open positions at end:", int(open_pos_series.iloc[-1]))
    print("Avg gross exposure %:", float(gross_exposure_series.mean() * 100.0))
    print("Max gross exposure %:", float(gross_exposure_series.max() * 100.0))
    print(f"Max DD% (MTM equity): {mtm_dd*100:.2f}%")
    print(f"Max DD% (Closed trades only): {closed_dd*100:.2f}%")

    # ============================================================
    # FTMO DAILY LOSS / INTRADAY DD ANALYSIS
    # ============================================================

    intraday_dd_df = compute_intraday_drawdowns(equity_series)

    daily_loss_stats = daily_loss_statistics(
        intraday_dd_df,
        soft_limit=-0.05,
        hard_limit=-0.10,
    )

    print("\n--- FTMO DAILY LOSS ANALYSIS (MTM, 1H) ---")
    for k, v in daily_loss_stats.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    # ============================================================
    # FTMO PIPELINE: bootstrap med -4% cutoff och högre eval exposure
    # ============================================================

    day_lib = build_day_path_library(
        equity_1h=equity_series,
        trades_df=portfolio_trades,  # trading day = minst 1 entry
    )

    params2 = FTMOParams2(
        challenge_target=CHALLENGE_TARGET,
        verification_target=VERIFICATION_TARGET,
        daily_loss_limit=DAILY_LOSS_LIMIT,
        max_loss_limit=MAX_LOSS_LIMIT,
        min_trading_days_eval=MIN_TRADING_DAYS_EVAL,
        payout_wait_days=PAYOUT_WAIT_DAYS,
        soft_cutoff=SOFT_CUTOFF_DAILY,
        eval_exposure_factor=EVAL_EXPOSURE_FACTOR,
        funded_exposure_factor=FUNDED_EXPOSURE_FACTOR,
    )

    sim2 = run_bootstrap_ftmo_pipeline_daypaths(
        day_lib=day_lib,
        start_balance=START_CAPITAL,
        n_iter=BOOT_N_ITER,
        horizon_days=BOOT_HORIZON_DAYS,
        block_len=BOOT_BLOCK_LEN,
        seed=BOOT_SEED,
        params=params2,
    )

    p_ch = float(sim2["passed_challenge"].mean())
    p_ver = float(sim2["passed_verification"].mean())
    p_pay = float(sim2["reached_payout"].mean())

    print("\n--- FTMO PIPELINE (DayPath Bootstrap) ---")
    print(
        f"Cutoff: {SOFT_CUTOFF_DAILY:.2%}, Eval exposure factor: {EVAL_EXPOSURE_FACTOR:.2f}, Funded factor: {FUNDED_EXPOSURE_FACTOR:.2f}")
    print(f"P(Pass Challenge):   {p_ch:.4f}")
    print(f"P(Pass Verification):{p_ver:.4f}")
    print(f"P(Reach 1st payout): {p_pay:.4f}")

    print("\n--- TIME TO EVENT (days) ---")
    print("Challenge:", summarize_times(sim2[sim2["passed_challenge"]], "days_to_challenge"))
    print("Verification:", summarize_times(sim2[sim2["passed_verification"]], "days_to_verification"))
    print("Payout:", summarize_times(sim2[sim2["reached_payout"]], "days_to_payout"))

    print("\n--- FAIL STAGE COUNTS ---")
    print(sim2["fail_stage"].value_counts(dropna=False))

    print("\n--- FAIL REASON COUNTS (non-null) ---")
    print(sim2["fail_reason"].dropna().value_counts())

    # 7) Plots
    plt.figure(figsize=(12, 5))
    plt.plot(equity_series.index, equity_series.values, label="MTM Equity")
    plt.plot(realized_equity_series.index, realized_equity_series.values, label="Realized (Closed only)", alpha=0.8)
    plt.title("Combined Portfolio Equity (TF 1H + MR Daily aligned to 1H)")
    plt.xlabel("Time")
    plt.ylabel("Equity ($)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 3))
    plt.plot(open_pos_series.index, open_pos_series.values)
    plt.title("Open Positions (market,strategy)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 3))
    plt.plot(gross_exposure_series.index, gross_exposure_series.values)
    plt.title("Gross Exposure % (Total)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Histogram över intraday DD
    plt.figure(figsize=(10, 4))
    plt.hist(intraday_dd_df["MinIntradayDD"], bins=50)
    plt.axvline(-0.05, color="orange", linestyle="--", label="-5%")
    plt.axvline(-0.10, color="red", linestyle="--", label="-10%")
    plt.title("Worst Intraday Drawdown per Day (MTM)")
    plt.xlabel("Intraday Drawdown")
    plt.ylabel("Days")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Visa värsta dagarna
    print("\n--- WORST INTRADAY DD DAYS ---")
    print(
        intraday_dd_df
        .sort_values("MinIntradayDD")
        .head(10)
    )
if __name__ == "__main__":
    main()