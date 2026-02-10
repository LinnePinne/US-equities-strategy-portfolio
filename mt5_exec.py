import time
import csv
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple, Dict

import MetaTrader5 as mt5


# ==========================
# CONFIG / DATA STRUCTURES
# ==========================

@dataclass
class ExecConfig:
    deviation_points: int = 50
    order_filling: int = mt5.ORDER_FILLING_FOK  # or mt5.ORDER_FILLING_IOC depending broker
    retries: int = 3
    retry_sleep_sec: float = 0.4
    min_margin_level: float = 200.0   # % (very conservative; FTMO usually huge headroom)
    log_csv_path: str = "trade_log.csv"


@dataclass
class SymbolMeta:
    name: str
    contract_size: float
    tick_size: float
    tick_value: float
    vol_min: float
    vol_max: float
    vol_step: float


# ==========================
# INIT / LOGGING
# ==========================

def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_initialized() -> None:
    if not mt5.initialize():
        raise RuntimeError("MT5 initialize() failed")
    acc = mt5.account_info()
    if acc is None:
        raise RuntimeError("MT5 account_info() failed (not logged in?)")


def ensure_symbol(symbol: str) -> SymbolMeta:
    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"Could not select symbol: {symbol}")

    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"symbol_info returned None for: {symbol}")

    return SymbolMeta(
        name=symbol,
        contract_size=float(info.trade_contract_size),
        tick_size=float(info.trade_tick_size),
        tick_value=float(info.trade_tick_value),
        vol_min=float(info.volume_min),
        vol_max=float(info.volume_max),
        vol_step=float(info.volume_step),
    )


def dollars_per_point_per_lot(meta: SymbolMeta) -> float:
    # $/point/lot = tick_value / tick_size  (here tick_size == point typically)
    if meta.tick_size <= 0:
        raise RuntimeError(f"Bad tick_size for {meta.name}: {meta.tick_size}")
    return meta.tick_value / meta.tick_size


def append_trade_log(cfg: ExecConfig, row: Dict) -> None:
    file_exists = False
    try:
        with open(cfg.log_csv_path, "r", encoding="utf-8") as _:
            file_exists = True
    except FileNotFoundError:
        file_exists = False

    fieldnames = list(row.keys())
    with open(cfg.log_csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerow(row)


# ==========================
# POSITION HELPERS
# ==========================

def get_position(symbol: str, magic: int):
    pos = mt5.positions_get(symbol=symbol)
    if pos is None:
        return None
    for p in pos:
        if int(p.magic) == int(magic):
            return p
    return None


def close_position_market(symbol: str, magic: int, cfg: ExecConfig, comment: str = "") -> bool:
    p = get_position(symbol, magic)
    if p is None:
        return True  # nothing to do

    volume = float(p.volume)
    # Long-only system: close by SELL
    return send_market_order(symbol, "SELL", volume, magic, cfg, comment=comment or "CLOSE")


# ==========================
# SIZING
# ==========================

def round_volume(meta: SymbolMeta, vol: float) -> float:
    vol = max(meta.vol_min, min(meta.vol_max, vol))
    step = meta.vol_step
    if step <= 0:
        return float(vol)
    steps = round(vol / step)
    return float(steps * step)


def notional_to_volume(meta: SymbolMeta, notional_usd: float, mid_price: float) -> float:
    """
    Your model: notional = price * contract_size * volume
    => volume = notional / (price * contract_size)
    """
    denom = mid_price * meta.contract_size
    if denom <= 0:
        return 0.0
    vol = notional_usd / denom
    return round_volume(meta, vol)


# ==========================
# RISK / ACCOUNT CHECKS
# ==========================

def account_snapshot() -> Dict[str, float]:
    acc = mt5.account_info()
    if acc is None:
        raise RuntimeError("account_info failed")
    return {
        "equity": float(acc.equity),
        "balance": float(acc.balance),
        "margin": float(acc.margin),
        "margin_free": float(acc.margin_free),
        "margin_level": float(acc.margin_level) if acc.margin_level is not None else 0.0,
    }


def can_open_new_trade(cfg: ExecConfig) -> Tuple[bool, str]:
    s = account_snapshot()
    ml = s["margin_level"]
    if ml > 0 and ml < cfg.min_margin_level:
        return False, f"Margin level too low: {ml:.1f}% < {cfg.min_margin_level:.1f}%"
    return True, "OK"


# ==========================
# ORDER SENDING (RETRY + ROBUST)
# ==========================

def _order_type(side: str) -> int:
    side = side.upper()
    if side == "BUY":
        return mt5.ORDER_TYPE_BUY
    if side == "SELL":
        return mt5.ORDER_TYPE_SELL
    raise ValueError("side must be BUY/SELL")


def _price_for_side(symbol: str, side: str) -> float:
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"symbol_info_tick failed for {symbol}")
    return float(tick.ask) if side.upper() == "BUY" else float(tick.bid)


def send_market_order(
    symbol: str,
    side: str,
    volume: float,
    magic: int,
    cfg: ExecConfig,
    comment: str = "",
) -> bool:
    side = side.upper()
    if volume <= 0:
        return False

    ok, reason = can_open_new_trade(cfg)
    if not ok and side == "BUY":
        # for SELL (closing) we allow even if margin is low
        return False

    meta = ensure_symbol(symbol)
    volume = round_volume(meta, float(volume))
    if volume < meta.vol_min:
        return False

    last_err = None

    for attempt in range(1, cfg.retries + 1):
        try:
            price = _price_for_side(symbol, side)

            req = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": float(volume),
                "type": _order_type(side),
                "price": float(price),
                "deviation": int(cfg.deviation_points),
                "magic": int(magic),
                "comment": comment[:31] if comment else "",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": int(cfg.order_filling),
            }

            res = mt5.order_send(req)
            if res is None:
                last_err = "order_send returned None"
            else:
                if res.retcode == mt5.TRADE_RETCODE_DONE:
                    append_trade_log(cfg, {
                        "ts": _ts(),
                        "symbol": symbol,
                        "side": side,
                        "volume": volume,
                        "magic": magic,
                        "comment": comment,
                        "retcode": res.retcode,
                        "order": res.order,
                        "deal": res.deal,
                        "price": price,
                    })
                    return True

                last_err = f"retcode={res.retcode} comment={getattr(res, 'comment', '')}"

        except Exception as e:
            last_err = str(e)

        time.sleep(cfg.retry_sleep_sec * attempt)

    # log failure
    append_trade_log(cfg, {
        "ts": _ts(),
        "symbol": symbol,
        "side": side,
        "volume": volume,
        "magic": magic,
        "comment": comment,
        "retcode": "FAIL",
        "order": "",
        "deal": "",
        "price": "",
        "error": last_err or "",
    })
    return False


# ==========================
# HIGH-LEVEL ENTRY FUNCTION
# ==========================

def open_long_by_notional(
    symbol: str,
    notional_usd: float,
    magic: int,
    cfg: ExecConfig,
    comment: str,
) -> Tuple[bool, float]:
    """
    Converts notional -> volume and submits BUY.
    Returns (success, volume_used).
    """
    meta = ensure_symbol(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return False, 0.0

    mid = (float(tick.bid) + float(tick.ask)) / 2.0
    vol = notional_to_volume(meta, float(notional_usd), mid)

    if vol <= 0:
        return False, 0.0

    ok = send_market_order(symbol, "BUY", vol, magic, cfg, comment=comment)
    return ok, vol
