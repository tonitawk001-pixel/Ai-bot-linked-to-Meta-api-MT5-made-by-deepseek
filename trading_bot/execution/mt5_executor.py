"""
MT5 Execution Engine — LIVE TRADE EXECUTION MODULE.

Handles order placement across multiple MT5 accounts with full safety validation.
Must NEVER bypass the Risk Engine (FINAL AUTHORITY).

Supports risk scoring: lot size is scaled by risk_evaluation.adjusted_lot_scale.
"""

import time
from datetime import datetime
from typing import Optional

import MetaTrader5 as mt5

from trading_bot.config import Config
from trading_bot.utils.logger import logger


def execute_trade(action: str, symbol: str, lot_size: float, sl: float, tp: float,
                  ohlcv=None, risk_evaluation: Optional[dict] = None,
                  account_list: Optional[list] = None) -> list:
    """
    Execute a trade across all configured MT5 accounts.

    Args:
        action: "BUY" or "SELL".
        symbol: MT5 symbol.
        lot_size: Base lot size (will be scaled by risk_evaluation.adjusted_lot_scale).
        sl: Stop loss price.
        tp: Take profit price.
        ohlcv: Optional OHLCV DataFrame for spread validation.
        risk_evaluation: Dict from risk_manager.validate().
        account_list: List of account dicts.

    Returns:
        list of dicts: Execution results per account.
    """
    if Config.EMERGENCY_STOP:
        logger.critical("EMERGENCY STOP ACTIVE — all execution blocked.")
        return [{"account": "ALL", "success": False, "reason": "Emergency stop active"}]

    if not Config.EXECUTION_ENABLED:
        logger.warning("EXECUTION_ENABLED is false. No trades placed.")
        return [{"account": "ALL", "success": False, "reason": "Execution disabled"}]

    if action.upper() not in ("BUY", "SELL"):
        logger.error(f"Invalid action '{action}'.")
        return [{"account": "ALL", "success": False, "reason": f"Invalid action {action}"}]

    # Risk evaluation check — must be approved (score < 70)
    if risk_evaluation and not risk_evaluation.get("approved", False):
        logger.warning(f"Risk engine BLOCKED trade: {risk_evaluation.get('reason', 'Unknown')}")
        return [{"account": "ALL", "success": False,
                 "reason": f"Risk block: {risk_evaluation.get('reason', 'Unknown')}"}]

    # Apply lot size scaling from risk scoring
    adjusted_lot_scale = 1.0
    if risk_evaluation:
        adjusted_lot_scale = risk_evaluation.get("adjusted_lot_scale", 1.0)
    final_lot = round(lot_size * adjusted_lot_scale, 2)
    final_lot = max(Config.MIN_LOT_SIZE, min(final_lot, Config.MAX_LOT_SIZE))

    if adjusted_lot_scale < 1.0:
        logger.info(f"Risk scaling applied: {lot_size} * {adjusted_lot_scale} = {final_lot}")

    # Pre-execution safety checks
    safety_issues = _pre_execution_checks(symbol, action, final_lot, ohlcv)
    if safety_issues:
        return [{"account": "ALL", "success": False, "reason": safety_issues}]

    accounts = account_list if account_list else Config.ACCOUNTS
    if not accounts:
        accounts = [{"login": Config.MT5_LOGIN, "password": Config.MT5_PASSWORD,
                     "server": Config.MT5_SERVER}]

    results = []
    order_type = mt5.ORDER_TYPE_BUY if action.upper() == "BUY" else mt5.ORDER_TYPE_SELL

    for acc in accounts:
        entry = {
            "account": f"{acc.get('login', '?')}@{acc.get('server', '?')}",
            "success": False, "reason": "", "order_ticket": None,
            "timestamp": datetime.now().isoformat(),
        }
        try:
            login = int(acc["login"])
            password = acc["password"]
            server = acc["server"]
            if not mt5.login(login, password, server):
                entry["reason"] = f"Login failed: {mt5.last_error()}"
                logger.error(f"Account {login}: {entry['reason']}")
                results.append(entry)
                continue

            info = mt5.symbol_info(symbol)
            if info is None:
                entry["reason"] = f"Symbol {symbol} not found"
                results.append(entry)
                continue
            if not info.visible:
                mt5.symbol_select(symbol, True)
                info = mt5.symbol_info(symbol)
            if info is None:
                entry["reason"] = f"Symbol {symbol} unavailable"
                results.append(entry)
                continue

            lot = max(info.volume_min, min(final_lot, info.volume_max))
            lot = round(lot, 2)

            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                entry["reason"] = "Cannot get tick data"
                results.append(entry)
                continue

            price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
            sl_price = round(sl, info.digits) if sl else 0.0
            tp_price = round(tp, info.digits) if tp else 0.0

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot,
                "type": order_type,
                "price": price,
                "sl": sl_price, "tp": tp_price,
                "deviation": 10, "magic": 123456,
                "comment": "AI_BOT_EXEC",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)
            if result is None:
                entry["reason"] = f"Order send failed: {mt5.last_error()}"
                results.append(entry)
                continue
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                entry["reason"] = f"Order retcode: {result.retcode}"
                results.append(entry)
                continue

            entry["success"] = True
            entry["reason"] = "Executed"
            entry["order_ticket"] = result.order
            entry["lot_size"] = lot
            entry["price"] = price
            logger.info(f"Account {login}: {action} {lot} {symbol} @ {price} "
                        f"SL={sl_price} TP={tp_price} (ticket #{result.order})")

        except Exception as exc:
            entry["reason"] = f"Exception: {exc}"
            logger.error(f"Account {acc.get('login')}: {exc}")

        results.append(entry)

    return results


def close_position(ticket: int, account: dict = None) -> dict:
    if Config.EMERGENCY_STOP:
        return {"success": False, "reason": "Emergency stop active"}
    try:
        if account:
            mt5.login(int(account["login"]), account["password"], account["server"])
        position = mt5.positions_get(ticket=ticket)
        if position is None or len(position) == 0:
            return {"success": False, "reason": f"Position {ticket} not found"}
        pos = position[0]
        order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(pos.symbol)
        if tick is None:
            return {"success": False, "reason": "Cannot get tick"}
        price = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol, "volume": pos.volume,
            "type": order_type, "position": ticket,
            "price": price, "deviation": 10, "magic": 123456,
            "comment": "AI_BOT_CLOSE",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result is None:
            return {"success": False, "reason": f"Close failed: {mt5.last_error()}"}
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return {"success": False, "reason": f"Close retcode: {result.retcode}"}
        return {"success": True, "reason": "Closed", "ticket": ticket, "price": price}
    except Exception as exc:
        return {"success": False, "reason": f"Exception: {exc}"}


def modify_position(ticket: int, sl: float = None, tp: float = None,
                    account: dict = None) -> dict:
    if Config.EMERGENCY_STOP:
        return {"success": False, "reason": "Emergency stop active"}
    try:
        if account:
            mt5.login(int(account["login"]), account["password"], account["server"])
        position = mt5.positions_get(ticket=ticket)
        if position is None or len(position) == 0:
            return {"success": False, "reason": f"Position {ticket} not found"}
        pos = position[0]
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": pos.symbol, "position": ticket,
            "sl": sl if sl else pos.sl,
            "tp": tp if tp else pos.tp,
        }
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            return {"success": False, "reason": f"Modify failed: {getattr(result, 'retcode', mt5.last_error())}"}
        return {"success": True, "reason": "Modified", "ticket": ticket}
    except Exception as exc:
        return {"success": False, "reason": f"Exception: {exc}"}


def _pre_execution_checks(symbol: str, action: str, lot_size: float,
                          ohlcv=None) -> Optional[str]:
    if lot_size < Config.MIN_LOT_SIZE or lot_size > Config.MAX_LOT_SIZE:
        return f"Lot size {lot_size} out of range [{Config.MIN_LOT_SIZE}-{Config.MAX_LOT_SIZE}]"
    info = mt5.symbol_info(symbol)
    if info is None:
        return f"Symbol {symbol} unknown"
    if not info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL or not info.visible:
        return f"Symbol {symbol} not tradable"
    if not info.session_deals > 0:
        return f"Market closed for {symbol}"
    if ohlcv is not None and "spread" in ohlcv.columns and len(ohlcv) > 5:
        spreads = ohlcv["spread"].iloc[-10:]
        avg_spread = spreads.mean()
        current_spread = ohlcv["spread"].iloc[-1]
        if current_spread > avg_spread * Config.MAX_SPREAD_MULTIPLIER:
            return f"Spread {current_spread} > {avg_spread * Config.MAX_SPREAD_MULTIPLIER:.0f}"
    return None