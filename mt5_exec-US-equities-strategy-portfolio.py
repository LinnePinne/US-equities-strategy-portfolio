import time
import csv
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple, Dict

import MetaTrader5 as mt5


# ==========================
# CONFIG
# ==========================

@dataclass
class ExecConfig:
    deviation_points: int = 50
    order_filling: int = mt5.ORDER_FILLING_IOC
    retries: int = 5
    retry_sleep_sec: float = 0.5
    min_margin_level: float = 150.0
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
# UTILS
# ==========================

def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_initialized():
    if not mt5.initialize():
        raise RuntimeError("MT5 initialize() failed")

    acc = mt5.account_info()
    if acc is None:
        raise RuntimeError("MT5 account_info() failed")


def ensure_symbol(symbol: str) -> SymbolMeta:
    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"Cannot select symbol: {symbol}")

    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"symbol_info None: {symbol}")

    return SymbolMeta(
        name=symbol,
        contract_size=float(info.trade_contract_size),
        tick_size=float(info.trade_tick_size),
        tick_value=float(info.trade_tick_value),
        vol_min=float(info.volume_min),
        vol_max=float(info.volume_max),
        vol_step=float(info.volume_step),
    )


# ==========================
# ACCOUNT
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
        "margin_level": float(acc.margin_level) if acc.margin_level else 0.0,
    }


# ==========================
# LOGGING
# ==========================

def append_trade_log(cfg: ExecConfig, row: Dict):

    file_exists = False
    try:
        with open(cfg.log_csv_path, "r"):
            file_exists = True
    except:
        pass

    with open(cfg.log_csv_path, "a", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(f, fieldnames=row.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


# ==========================
# POSITION HELPERS
# ==========================

def get_positions(symbol: Optional[str] = None):

    if symbol:
        return mt5.positions_get(symbol=symbol)
    return mt5.positions_get()


def get_position(symbol: str, magic: int):

    pos = mt5.positions_get(symbol=symbol)

    if pos is None:
        return None

    for p in pos:
        if int(p.magic) == int(magic):
            return p

    return None


# ==========================
# SIZING
# ==========================

def round_volume(meta: SymbolMeta, volume: float) -> float:

    volume = max(meta.vol_min, min(meta.vol_max, volume))

    step = meta.vol_step
    if step <= 0:
        return volume

    steps = round(volume / step)

    return steps * step


def notional_to_volume(meta: SymbolMeta, notional_usd: float, price: float) -> float:

    denom = price * meta.contract_size

    if denom <= 0:
        return 0.0

    volume = notional_usd / denom

    return round_volume(meta, volume)


# ==========================
# ORDER CORE
# ==========================

def _order_type(side: str):

    if side == "BUY":
        return mt5.ORDER_TYPE_BUY
    if side == "SELL":
        return mt5.ORDER_TYPE_SELL

    raise ValueError("side must be BUY/SELL")


def _price(symbol: str, side: str):

    tick = mt5.symbol_info_tick(symbol)

    if tick is None:
        raise RuntimeError("No tick")

    if side == "BUY":
        return float(tick.ask)

    return float(tick.bid)


def send_market_order(
    symbol: str,
    side: str,
    volume: float,
    magic: int,
    cfg: ExecConfig,
    comment: str = "",
    position_ticket: Optional[int] = None,
) -> bool:

    side = side.upper()

    meta = ensure_symbol(symbol)

    volume = round_volume(meta, volume)

    if volume <= 0:
        return False

    for attempt in range(cfg.retries):

        try:

            price = _price(symbol, side)

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": _order_type(side),
                "price": price,
                "deviation": cfg.deviation_points,
                "magic": magic,
                "comment": comment[:30],
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": cfg.order_filling,
            }

            # critical for hedging accounts
            if position_ticket is not None:
                request["position"] = int(position_ticket)

            result = mt5.order_send(request)

            if result is None:
                continue

            if result.retcode == mt5.TRADE_RETCODE_DONE:

                append_trade_log(cfg, {
                    "ts": _ts(),
                    "symbol": symbol,
                    "side": side,
                    "volume": volume,
                    "magic": magic,
                    "ticket": position_ticket,
                    "comment": comment,
                    "price": price,
                    "retcode": result.retcode,
                })

                return True

        except Exception:
            pass

        time.sleep(cfg.retry_sleep_sec)

    return False


# ==========================
# CLOSE POSITION (SAFE)
# ==========================

def close_position_market(
    symbol: str,
    magic: int,
    cfg: ExecConfig,
    comment: str = "CLOSE"
) -> bool:

    p = get_position(symbol, magic)

    if p is None:
        return True

    volume = float(p.volume)

    # detect direction
    if p.type == mt5.POSITION_TYPE_BUY:
        side = "SELL"
    else:
        side = "BUY"

    ticket = int(p.ticket)

    ok = send_market_order(
        symbol=symbol,
        side=side,
        volume=volume,
        magic=magic,
        cfg=cfg,
        comment=comment,
        position_ticket=ticket,
    )

    return ok


# ==========================
# PANIC CLOSE ALL
# ==========================

def close_all_positions(cfg: ExecConfig):

    positions = mt5.positions_get()

    if positions is None:
        return

    for p in positions:

        symbol = p.symbol
        volume = float(p.volume)
        ticket = int(p.ticket)

        if p.type == mt5.POSITION_TYPE_BUY:
            side = "SELL"
        else:
            side = "BUY"

        send_market_order(
            symbol=symbol,
            side=side,
            volume=volume,
            magic=int(p.magic),
            cfg=cfg,
            comment="PANIC_CLOSE",
            position_ticket=ticket,
        )


# ==========================
# ENTRY
# ==========================

def open_long_by_notional(
    symbol: str,
    notional_usd: float,
    magic: int,
    cfg: ExecConfig,
    comment: str = ""
) -> Tuple[bool, float]:

    meta = ensure_symbol(symbol)

    tick = mt5.symbol_info_tick(symbol)

    if tick is None:
        return False, 0.0

    price = (float(tick.bid) + float(tick.ask)) / 2.0

    volume = notional_to_volume(meta, notional_usd, price)

    if volume <= 0:
        return False, 0.0

    ok = send_market_order(
        symbol=symbol,
        side="BUY",
        volume=volume,
        magic=magic,
        cfg=cfg,
        comment=comment,
    )

    return ok, volume
