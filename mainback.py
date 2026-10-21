import os
import time
import math
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import uvicorn

# =====================================================================
# 1. 核心策略参数配置 (15m Supertrend 触线即买激进版)
# =====================================================================
CONFIG = {
    "initial_capital": 10000.0,   # 初始模拟本金 (USDT)
    "leverage": 3,                # 杠杆倍数 (3x)
    "position_pct": 0.10,         # 单笔仓位占总权益比例 (10%)
    "tp_pct": 0.10,               # 止盈比例 (+10% 价格涨幅，限价挂单)
    "sl_pct": 0.05,               # 止损比例 (-5% 价格跌幅，条件市价单)
    "maker_fee": 0.0002,          # 限价单手续费 (0.02% - 用于触线限价开仓 & 限价止盈)
    "taker_fee": 0.0005,          # 市价单手续费 (0.05% - 用于市价止损)
    "st_period": 10,              # Supertrend ATR 周期
    "st_factor": 3.0,             # Supertrend 倍数
    "klines_limit": 200,          # 拉取 200 根 K 线确保 RMA 完全收敛
    "poll_interval": 3,           # 实时价格与触线监控轮询间隔 (秒)
    "db_path": "trading_v5.db",   # SQLite 数据库文件路径
}

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ERAUSDT"]

# =====================================================================
# 2. Supertrend 指标计算 (修复 NaN 传导，对齐 TradingView RMA)
# =====================================================================
def calculate_supertrend(df: pd.DataFrame, period: int = 10, factor: float = 3.0) -> pd.DataFrame:
    high, low, close = df['high'], df['low'], df['close']
    hl = high - low
    hc = (high - close.shift(1)).abs()
    lc = (low - close.shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)

    # Wilder's RMA 指数平滑
    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    hl2 = (high + low) / 2.0
    basic_upper = hl2 + (factor * atr)
    basic_lower = hl2 - (factor * atr)

    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    st_val = pd.Series(index=df.index, dtype=float)
    st_dir = pd.Series(index=df.index, dtype=int)

    start_idx = period
    if start_idx >= len(df):
        df['st_val'] = np.nan
        df['st_dir'] = 1
        return df

    st_dir.iloc[:start_idx] = 1

    for i in range(start_idx, len(df)):
        # 下轨只升不降
        if pd.isna(final_lower.iloc[i - 1]) or (basic_lower.iloc[i] > final_lower.iloc[i - 1]) or (close.iloc[i - 1] < final_lower.iloc[i - 1]):
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]

        # 上轨只降不升
        if pd.isna(final_upper.iloc[i - 1]) or (basic_upper.iloc[i] < final_upper.iloc[i - 1]) or (close.iloc[i - 1] > final_upper.iloc[i - 1]):
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]

        prev_dir = st_dir.iloc[i - 1]
        if prev_dir == 1 and close.iloc[i] > final_upper.iloc[i - 1]:
            current_dir = -1  # 翻多
        elif prev_dir == -1 and close.iloc[i] < final_lower.iloc[i - 1]:
            current_dir = 1   # 翻空
        else:
            current_dir = prev_dir

        st_dir.iloc[i] = current_dir
        st_val.iloc[i] = final_lower.iloc[i] if current_dir == -1 else final_upper.iloc[i]

    df['st_val'] = st_val
    df['st_dir'] = st_dir
    return df


# =====================================================================
# 3. 币安合约交易规则缓存 (精度截断与最小名义价值校验)
# =====================================================================
class ExchangeRules:
    def __init__(self):
        self.rules: Dict[str, dict] = {}
        self.refresh()

    def refresh(self):
        try:
            url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
            res = requests.get(url, timeout=10).json()
            for s in res.get("symbols", []):
                if s.get("contractType") == "PERPETUAL" and s.get("quoteAsset") == "USDT":
                    step_size = 0.001
                    tick_size = 0.0001
                    min_notional = 5.0
                    for f in s.get("filters", []):
                        if f["filterType"] == "LOT_SIZE":
                            step_size = float(f["stepSize"])
                        elif f["filterType"] == "priceFilter":
                            tick_size = float(f["tickSize"])
                        elif f["filterType"] == "MIN_NOTIONAL":
                            min_notional = float(f.get("notional", 5.0))
                    self.rules[s["symbol"]] = {
                        "step_size": step_size,
                        "tick_size": tick_size,
                        "min_notional": min_notional
                    }
        except Exception as e:
            print(f"[!] 获取币安交易规则失败 (将使用默认精度): {e}")

    def round_qty(self, symbol: str, qty: float) -> float:
        step = self.rules.get(symbol, {}).get("step_size", 0.001)
        if step <= 0:
            return round(qty, 3)
        precision = max(0, int(round(-math.log10(step))))
        return round(math.floor(qty / step) * step, precision)

    def round_price(self, symbol: str, price: float) -> float:
        tick = self.rules.get(symbol, {}).get("tick_size", 0.0001)
        if tick <= 0:
            return round(price, 4)
        precision = max(0, int(round(-math.log10(tick))))
        return round(round(price / tick) * tick, precision)

    def get_min_notional(self, symbol: str) -> float:
        return self.rules.get(symbol, {}).get("min_notional", 5.0)


# =====================================================================
# 4. SQLite 持久化数据库管理
# =====================================================================
class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_db()

    def _get_conn(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_db(self):
        with self.lock, self._get_conn() as conn:
            c = conn.cursor()
            c.execute("CREATE TABLE IF NOT EXISTS account (id INTEGER PRIMARY KEY, balance REAL)")
            c.execute("SELECT balance FROM account WHERE id = 1")
            if not c.fetchone():
                c.execute("INSERT INTO account (id, balance) VALUES (1, ?)", (CONFIG["initial_capital"],))

            c.execute("""CREATE TABLE IF NOT EXISTS watch_symbols (
                symbol TEXT PRIMARY KEY
            )""")
            c.execute("SELECT COUNT(*) FROM watch_symbols")
            if c.fetchone()[0] == 0:
                for s in DEFAULT_SYMBOLS:
                    c.execute("INSERT OR IGNORE INTO watch_symbols (symbol) VALUES (?)", (s,))

            c.execute("""CREATE TABLE IF NOT EXISTS positions (
                symbol TEXT PRIMARY KEY,
                entry_price REAL,
                qty REAL,
                margin REAL,
                tp_price REAL,
                sl_price REAL,
                entry_time TEXT,
                entry_time_ms INTEGER,
                bar_open_time INTEGER
            )""")

            c.execute("""CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                entry_price REAL,
                exit_price REAL,
                qty REAL,
                pnl REAL,
                pnl_pct REAL,
                reason TEXT,
                entry_time TEXT,
                exit_time TEXT
            )""")
            conn.commit()

    def get_balance(self) -> float:
        with self.lock, self._get_conn() as conn:
            return float(conn.execute("SELECT balance FROM account WHERE id = 1").fetchone()[0])

    def update_balance(self, new_balance: float):
        with self.lock, self._get_conn() as conn:
            conn.execute("UPDATE account SET balance = ? WHERE id = 1", (new_balance,))
            conn.commit()

    def get_symbols(self) -> List[str]:
        with self.lock, self._get_conn() as conn:
            rows = conn.execute("SELECT symbol FROM watch_symbols ORDER BY symbol").fetchall()
            return [r[0] for r in rows]

    def add_symbol(self, symbol: str):
        with self.lock, self._get_conn() as conn:
            conn.execute("INSERT OR IGNORE INTO watch_symbols (symbol) VALUES (?)", (symbol,))
            conn.commit()

    def remove_symbol(self, symbol: str):
        with self.lock, self._get_conn() as conn:
            conn.execute("DELETE FROM watch_symbols WHERE symbol = ?", (symbol,))
            conn.commit()

    def get_positions(self) -> Dict[str, dict]:
        with self.lock, self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM positions").fetchall()
            return {r["symbol"]: dict(r) for r in rows}

    def save_position(self, pos: dict):
        with self.lock, self._get_conn() as conn:
            conn.execute("""INSERT OR REPLACE INTO positions 
                (symbol, entry_price, qty, margin, tp_price, sl_price, entry_time, entry_time_ms, bar_open_time)
                VALUES (:symbol, :entry_price, :qty, :margin, :tp_price, :sl_price, :entry_time, :entry_time_ms, :bar_open_time)""", pos)
            conn.commit()

    def delete_position(self, symbol: str):
        with self.lock, self._get_conn() as conn:
            conn.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
            conn.commit()

    def add_trade(self, trade: dict):
        with self.lock, self._get_conn() as conn:
            conn.execute("""INSERT INTO trades 
                (symbol, entry_price, exit_price, qty, pnl, pnl_pct, reason, entry_time, exit_time)
                VALUES (:symbol, :entry_price, :exit_price, :qty, :pnl, :pnl_pct, :reason, :entry_time, :exit_time)""", trade)
            conn.commit()

    def get_trades(self, limit: int = 50) -> List[dict]:
        with self.lock, self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
            return [dict(r) for r in rows]


# =====================================================================
# 5. 交易核心引擎 (实时触线挂单买入 + 限价止盈 + 市价止损)
# =====================================================================
class TradingEngine:
    def __init__(self):
        self.cfg = CONFIG
        self.db = Database(self.cfg["db_path"])
        self.rules = ExchangeRules()
        self.positions: Dict[str, dict] = self.db.get_positions()
        # 缓存每个币种当前 15m 的多头支撑位与状态: {symbol: {"st_dir": -1, "st_val": 0.0706, "bar_open_time": 1729...}}
        self.active_supports: Dict[str, dict] = {}
        # 记录在某根 15m K线已经开过仓的标识，防止同一根K线止损后反复接针: {symbol: bar_open_time}
        self.triggered_bars: Dict[str, int] = {}
        self.latest_prices: Dict[str, float] = {}
        self.last_15m_Refresh: int = -1

    def get_watch_symbols(self) -> List[str]:
        return self.db.get_symbols()

    def fetch_klines(self, symbol: str, interval: str = "15m", limit: int = 200) -> Optional[pd.DataFrame]:
        try:
            url = "https://fapi.binance.com/fapi/v1/klines"
            params = {"symbol": symbol, "interval": interval, "limit": limit}
            res = requests.get(url, params=params, timeout=6)
            if res.status_code != 200:
                return None
            data = res.json()
            df = pd.DataFrame(data, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "qav", "num_trades", "taker_base", "taker_quote", "ignore"
            ])
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)
            df["open_time"] = df["open_time"].astype(int)
            df["close_time"] = df["close_time"].astype(int)
            return df
        except Exception as e:
            print(f"[!] 拉取 {symbol} K线异常: {e}")
            return None

    def fetch_all_prices(self) -> Dict[str, float]:
        try:
            url = "https://fapi.binance.com/fapi/v1/ticker/price"
            res = requests.get(url, timeout=5).json()
            prices = {item["symbol"]: float(item["price"]) for item in res}
            self.latest_prices = prices
            return prices
        except Exception:
            return self.latest_prices

    def refresh_supertrend_supports(self, symbols: Optional[List[str]] = None):
        """更新所有监控币种的 15m Supertrend 支撑位"""
        target_symbols = symbols if symbols is not None else self.get_watch_symbols()
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now_str}] >>> 刷新 15m Supertrend 轨道参数 (共 {len(target_symbols)} 个币)...")

        for sym in target_symbols:
            df = self.fetch_klines(sym, "15m", limit=self.cfg["klines_limit"])
            if df is None or len(df) < 30:
                continue
            df = calculate_supertrend(df, self.cfg["st_period"], self.cfg["st_factor"])
            
            # 倒数第二根是已完全收盘定型的 K 线，倒数第一根是当前正在走的 15m K 线
            closed_bar = df.iloc[-2]
            current_bar = df.iloc[-1]
            
            # 只要已收盘 K 线处于多头趋势 (st_dir == -1)，其支撑线将作为当前盘中触线买入的基准挂单线
            st_dir = int(closed_bar["st_dir"])
            st_val = float(closed_bar["st_val"]) if not pd.isna(closed_bar["st_val"]) else 0.0
            bar_open_time = int(current_bar["open_time"])

            self.active_supports[sym] = {
                "st_dir": st_dir,
                "st_val": self.rules.round_price(sym, st_val),
                "bar_open_time": bar_open_time,
                "last_close": float(closed_bar["close"])
            }

    def check_touch_entries_and_positions(self):
        """高频巡检：1. 实时触及支撑线立即限价买入；2. 持仓限价止盈与市价止损"""
        prices = self.fetch_all_prices()
        if not prices:
            return

        watch_symbols = self.get_watch_symbols()

        # -----------------------------------------------------------------
        # A. 检查未持仓币种是否触及多头支撑线 (Touch & Buy)
        # -----------------------------------------------------------------
        max_positions = int(1.0 / self.cfg["position_pct"])
        for sym in watch_symbols:
            if sym in self.positions:
                continue
            if len(self.positions) >= max_positions:
                break

            # 若新添加的币种还没有支撑线缓存，立即单独拉取初始化
            if sym not in self.active_supports:
                self.refresh_supertrend_supports([sym])

            sup = self.active_supports.get(sym)
            if not sup or sup["st_dir"] != -1 or sup["st_val"] <= 0:
                continue

            # 同根 15m K 线防重复开仓保护
            if self.triggered_bars.get(sym) == sup["bar_open_time"]:
                continue

            cur_price = prices.get(sym)
            if not cur_price:
                continue

            support_price = sup["st_val"]

            # 当实时价已经回落接近支撑位 (+1.2% 以内) 或跌破支撑位时，核实 1m K线最低价防漏针
            low_touch = cur_price
            if cur_price <= support_price * 1.012:
                df_1m = self.fetch_klines(sym, "1m", limit=2)
                if df_1m is not None and not df_1m.empty:
                    valid_1m = df_1m[df_1m["close_time"] > sup["bar_open_time"]]
                    if not valid_1m.empty:
                        low_touch = min(cur_price, float(valid_1m["low"].min()))

            # 触发条件：盘中价格触及或刺穿多头支撑线，并且上一根收盘价在支撑线上方 (确保是回踩而非早已跌破)
            if low_touch <= support_price and sup["last_close"] >= support_price:
                # 防止当前价格已经暴跌击穿超过 -3% 的极端闪崩情况，若正常触线则按支撑位限价挂单价成交
                exec_entry_price = support_price if cur_price <= support_price * 1.005 else cur_price
                self.execute_entry(sym, exec_entry_price, sup["bar_open_time"])

        # -----------------------------------------------------------------
        # B. 检查已有持仓：限价止盈 (TP Limit) & 条件市价止损 (SL Market)
        # -----------------------------------------------------------------
        for sym, pos in list(self.positions.items()):
            cur_price = prices.get(sym)
            if not cur_price:
                continue

            tp_price = pos["tp_price"]
            sl_price = pos["sl_price"]
            entry_time_ms = pos["entry_time_ms"]

            high_price = cur_price
            low_price = cur_price

            # 当价格接近止盈或止损区时，拉取 1m K 线校验 High/Low 防漏秒针
            if cur_price >= tp_price * 0.985 or cur_price <= sl_price * 1.015:
                df_1m = self.fetch_klines(sym, "1m", limit=2)
                if df_1m is not None and not df_1m.empty:
                    recent = df_1m[df_1m["close_time"] > entry_time_ms]
                    if not recent.empty:
                        high_price = max(cur_price, float(recent["high"].max()))
                        low_price = min(cur_price, float(recent["low"].min()))

            # 1. 限价止盈挂单触发：严格按挂单限价 tp_price 零误差结算 (收 Maker 费率)
            if high_price >= tp_price:
                self.execute_exit(sym, tp_price, "止盈限价成交(TP Limit)", is_limit=True)

            # 2. 市价止损触发：按当前市价(若跳空则取更真实的市价，正常情况为 sl_price)结算 (收 Taker 费率)
            elif low_price <= sl_price:
                market_sl_exec = min(cur_price, sl_price)
                self.execute_exit(sym, market_sl_exec, "止损市价成交(SL Market)", is_limit=False)

    def execute_entry(self, symbol: str, entry_price: float, bar_open_time: int):
        balance = self.db.get_balance()
        # 计算总动态权益 = 可用现金 + 已占用保证金
        used_margin = sum(p["margin"] for p in self.positions.values())
        total_equity = balance + used_margin

        margin = total_equity * self.cfg["position_pct"]
        if margin > balance or margin <= 0:
            return

        notional = margin * self.cfg["leverage"]
        qty = self.rules.round_qty(symbol, notional / entry_price)
        entry_price = self.rules.round_price(symbol, entry_price)
        actual_notional = qty * entry_price

        if actual_notional < self.rules.get_min_notional(symbol) or qty <= 0:
            return

        actual_margin = actual_notional / self.cfg["leverage"]
        # 触线挂单买入属于 Maker 限价单
        entry_fee = actual_notional * self.cfg["maker_fee"]

        tp_price = self.rules.round_price(symbol, entry_price * (1.0 + self.cfg["tp_pct"]))
        sl_price = self.rules.round_price(symbol, entry_price * (1.0 - self.cfg["sl_pct"]))

        now_utc = datetime.now(timezone.utc)
        pos = {
            "symbol": symbol,
            "entry_price": entry_price,
            "qty": qty,
            "margin": round(actual_margin, 4),
            "tp_price": tp_price,
            "sl_price": sl_price,
            "entry_time": now_utc.strftime("%Y-%m-%d %H:%M:%S"),
            "entry_time_ms": int(now_utc.timestamp() * 1000),
            "bar_open_time": bar_open_time
        }

        self.db.update_balance(balance - actual_margin - entry_fee)
        self.db.save_position(pos)
        self.positions[symbol] = pos
        self.triggered_bars[symbol] = bar_open_time

        print(f"[★ 触线买入] {symbol} | 挂单入场价: {entry_price} | 数量: {qty} | 限价止盈: {tp_price} | 市价止损: {sl_price}")

    def execute_exit(self, symbol: str, exit_price: float, reason: str, is_limit: bool = True):
        pos = self.positions.get(symbol)
        if not pos:
            return

        exit_price = self.rules.round_price(symbol, exit_price)
        qty = pos["qty"]
        entry_price = pos["entry_price"]
        margin = pos["margin"]

        gross_pnl = (exit_price - entry_price) * qty
        fee_rate = self.cfg["maker_fee"] if is_limit else self.cfg["taker_fee"]
        exit_fee = (qty * exit_price) * fee_rate
        net_pnl = gross_pnl - exit_fee
        pnl_pct = (net_pnl / margin) * 100.0 if margin > 0 else 0.0

        balance = self.db.get_balance()
        self.db.update_balance(balance + margin + net_pnl)
        self.db.delete_position(symbol)
        del self.positions[symbol]

        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        trade = {
            "symbol": symbol,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "qty": qty,
            "pnl": round(net_pnl, 4),
            "pnl_pct": round(pnl_pct, 2),
            "reason": reason,
            "entry_time": pos["entry_time"],
            "exit_time": now_str
        }
        self.db.add_trade(trade)
        print(f"[■ 平仓结算] {symbol} | {reason} | 出场价: {exit_price} | 净盈亏: {net_pnl:+.2f} USDT ({pnl_pct:+.2f}%)")

    def run_forever(self):
        print("后台交易调度线程已启动 (模式: 15m Supertrend 触线即买 + 限价止盈/市价止损)...")
        self.refresh_supertrend_supports()
        while True:
            try:
                now_utc = datetime.now(timezone.utc)
                current_15m_block = (now_utc.hour * 60 + now_utc.minute) // 15

                # 每逢新的 15 分钟周期开启后 (第 2 秒后)，刷新全币池 Supertrend 支撑轨
                if current_15m_block != self.last_15m_Refresh and now_utc.second >= 2:
                    self.refresh_supertrend_supports()
                    self.last_15m_Refresh = current_15m_block

                # 每 3 秒高频巡检触线买入与止盈止损
                self.check_touch_entries_and_positions()
            except Exception as e:
                print(f"[!] 调度循环异常: {e}")
            time.sleep(self.cfg["poll_interval"])


# =====================================================================
# 6. FastAPI Web 控制台与接口
# =====================================================================
app = FastAPI(title="15m Supertrend Touch-Buy Engine")
engine = TradingEngine()

@app.on_event("startup")
def startup_event():
    t = threading.Thread(target=engine.run_forever, daemon=True)
    t.start()

@app.get("/api/state")
def api_state():
    balance = engine.db.get_balance()
    prices = engine.latest_prices
    pos_list = []
    unrealized_pnl = 0.0
    used_margin = 0.0

    for sym, p in engine.positions.items():
        cur = prices.get(sym, p["entry_price"])
        upnl = (cur - p["entry_price"]) * p["qty"]
        upnl_pct = (upnl / p["margin"]) * 100.0 if p["margin"] > 0 else 0.0
        unrealized_pnl += upnl
        used_margin += p["margin"]
        pos_list.append({
            **p,
            "current_price": cur,
            "upnl": round(upnl, 2),
            "upnl_pct": round(upnl_pct, 2)
        })

    watch_details = []
    for sym in engine.get_watch_symbols():
        sup = engine.active_supports.get(sym, {})
        watch_details.append({
            "symbol": sym,
            "price": prices.get(sym, 0.0),
            "st_dir": sup.get("st_dir", 0),
            "st_val": sup.get("st_val", 0.0)
        })

    return {
        "balance": round(balance, 2),
        "equity": round(balance + used_margin + unrealized_pnl, 2),
        "unrealized_pnl": round(unrealized_pnl, 2),
        "positions": pos_list,
        "watch_list": watch_details,
        "trades": engine.db.get_trades(30)
    }

@app.post("/api/symbols/add")
def api_add_symbol(payload: dict):
    s = payload.get("symbol", "").upper().strip()
    if not s or not s.endswith("USDT"):
        raise HTTPException(status_code=400, detail="币种格式需形如 BTCUSDT")
    if engine.rules.rules and (s not in engine.rules.rules):
        raise HTTPException(status_code=400, detail=f"未在币安永续合约中找到 {s}")
    engine.db.add_symbol(s)
    # 立即异步计算该新币的支撑线
    threading.Thread(target=engine.refresh_supertrend_supports, args=([s],), daemon=True).start()
    return {"status": "ok"}

@app.post("/api/symbols/remove")
def api_remove_symbol(payload: dict):
    s = payload.get("symbol", "").upper().strip()
    engine.db.remove_symbol(s)
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
def index_page():
    return """
<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <title>15m Supertrend 触线即买量化终端 v5.0</title>
    <style>
        body { background:#0d1117; color:#c9d1d9; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; margin:0; padding:24px; }
        .container { max-width:1200px; margin:0 auto; }
        h1 { font-size:22px; color:#58a6ff; margin-bottom:16px; display:flex; align-items:center; justify-content:space-between; }
        .cards { display:grid; grid-template-columns:repeat(3, 1fr); gap:16px; margin-bottom:24px; }
        .card { background:#161b22; border:1px solid #30363d; border-radius:8px; padding:16px; }
        .card-title { font-size:13px; color:#8b949e; margin-bottom:8px; }
        .card-val { font-size:24px; font-weight:bold; color:#f0f6fc; }
        .panel { background:#161b22; border:1px solid #30363d; border-radius:8px; padding:16px; margin-bottom:24px; }
        .panel-title { font-size:16px; font-weight:600; margin-bottom:14px; color:#f0f6fc; border-bottom:1px solid #21262d; padding-bottom:8px; }
        .sym-grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(220px, 1fr)); gap:10px; margin-top:12px; }
        .sym-box { background:#21262d; border:1px solid #30363d; border-radius:6px; padding:10px 12px; font-size:13px; position:relative; }
        .sym-box .del { position:absolute; right:10px; top:10px; cursor:pointer; color:#f85149; font-weight:bold; }
        input { background:#0d1117; border:1px solid #30363d; color:#fff; padding:8px 12px; border-radius:6px; margin-right:8px; }
        button { background:#238636; color:#fff; border:none; padding:8px 16px; border-radius:6px; cursor:pointer; font-weight:600; }
        button:hover { background:#2ea043; }
        table { width:100%; border-collapse:collapse; font-size:13px; }
        th, td { text-align:left; padding:10px 8px; border-bottom:1px solid #21262d; }
        th { color:#8b949e; }
        .pos { color:#3fb950; font-weight:bold; }
        .neg { color:#f85149; font-weight:bold; }
        .tag-bull { background:rgba(63,185,80,0.15); color:#3fb950; padding:2px 6px; border-radius:4px; font-size:12px; }
        .tag-bear { background:rgba(248,81,73,0.15); color:#f85149; padding:2px 6px; border-radius:4px; font-size:12px; }
    </style>
</head>
<body>
<div class="container">
    <h1>
        <span>⚡ 15m Supertrend 触线即买系统 v5.0 (限价止盈 / 市价止损)</span>
        <span style="font-size:13px;color:#8b949e;">自动巡检: 3秒/次</span>
    </h1>
    <div class="cards">
        <div class="card"><div class="card-title">账户总权益 (USDT)</div><div class="card-val" id="equity">--</div></div>
        <div class="card"><div class="card-title">可用现金余额 (USDT)</div><div class="card-val" id="balance">--</div></div>
        <div class="card"><div class="card-title">当前持仓浮动盈亏 (USDT)</div><div class="card-val" id="upnl">--</div></div>
    </div>

    <div class="panel">
        <div class="panel-title">雷达监控池 (显示实时价与 15m 支撑挂单线)</div>
        <div>
            <input type="text" id="symbol-input" placeholder="输入币种 如 ERAUSDT" />
            <button onclick="addSymbol()">+ 添加监控标的</button>
        </div>
        <div class="sym-grid" id="sym-grid"></div>
    </div>

    <div class="panel">
        <div class="panel-title">当前活跃持仓 (3x 杠杆)</div>
        <table>
            <thead><tr><th>币种</th><th>触线入场价</th><th>当前市价</th><th>数量</th><th>保证金</th><th>限价止盈(+10%)</th><th>市价止损(-5%)</th><th>浮动盈亏</th><th>开仓时间(UTC)</th></tr></thead>
            <tbody id="pos-body"></tbody>
        </table>
    </div>

    <div class="panel">
        <div class="panel-title">历史平仓账单</div>
        <table>
            <thead><tr><th>币种</th><th>入场价</th><th>平仓结算价</th><th>数量</th><th>净盈亏(USDT)</th><th>收益率</th><th>触发原因</th><th>平仓时间(UTC)</th></tr></thead>
            <tbody id="trade-body"></tbody>
        </table>
    </div>
</div>

<script>
async function fetchData() {
    try {
        const res = await fetch('/api/state');
        const d = await res.json();
        document.getElementById('equity').innerText = d.equity.toFixed(2);
        document.getElementById('balance').innerText = d.balance.toFixed(2);
        const upnlEl = document.getElementById('upnl');
        upnlEl.innerText = (d.unrealized_pnl >= 0 ? '+' : '') + d.unrealized_pnl.toFixed(2);
        upnlEl.className = 'card-val ' + (d.unrealized_pnl >= 0 ? 'pos' : 'neg');

        document.getElementById('sym-grid').innerHTML = d.watch_list.map(item => {
            const isBull = item.st_dir === -1;
            const badge = isBull ? '<span class="tag-bull">多头待接针</span>' : '<span class="tag-bear">空头观望</span>';
            return `<div class="sym-box">
                <span class="del" onclick="removeSymbol('${item.symbol}')">✕</span>
                <div style="font-weight:bold;color:#f0f6fc;margin-bottom:4px;">${item.symbol} ${badge}</div>
                <div>当前价: <b style="color:#58a6ff">${item.price || '--'}</b></div>
                <div>挂单线: <b>${item.st_val || '计算中'}</b></div>
            </div>`;
        }).join('');

        document.getElementById('pos-body').innerHTML = d.positions.length ? d.positions.map(p => `
            <tr>
                <td><b>${p.symbol}</b></td>
                <td>${p.entry_price}</td>
                <td>${p.current_price}</td>
                <td>${p.qty}</td>
                <td>${p.margin}</td>
                <td class="pos">${p.tp_price}</td>
                <td class="neg">${p.sl_price}</td>
                <td class="${p.upnl >= 0 ? 'pos' : 'neg'}">${p.upnl >= 0 ? '+' : ''}${p.upnl} (${p.upnl_pct}%)</td>
                <td>${p.entry_time}</td>
            </tr>`).join('') : '<tr><td colspan="9" style="text-align:center;color:#8b949e;">暂无持仓，等待价格回踩多头支撑线...</td></tr>';

        document.getElementById('trade-body').innerHTML = d.trades.length ? d.trades.map(t => `
            <tr>
                <td><b>${t.symbol}</b></td>
                <td>${t.entry_price}</td>
                <td>${t.exit_price}</td>
                <td>${t.qty}</td>
                <td class="${t.pnl >= 0 ? 'pos' : 'neg'}">${t.pnl >= 0 ? '+' : ''}${t.pnl}</td>
                <td class="${t.pnl >= 0 ? 'pos' : 'neg'}">${t.pnl_pct}%</td>
                <td>${t.reason}</td>
                <td>${t.exit_time}</td>
            </tr>`).join('') : '<tr><td colspan="8" style="text-align:center;color:#8b949e;">暂无历史交易记录</td></tr>';
    } catch (e) {}
}

async function addSymbol() {
    const input = document.getElementById('symbol-input');
    const symbol = input.value.trim().toUpperCase();
    if (!symbol) return;
    try {
        const res = await fetch('/api/symbols/add', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({symbol})
        });
        const d = await res.json();
        if (!res.ok) { alert('添加失败: ' + (d.detail || res.statusText)); return; }
        input.value = '';
        setTimeout(fetchData, 600);
    } catch (e) { alert('网络请求异常: ' + e.message); }
}

async function removeSymbol(symbol) {
    await fetch('/api/symbols/remove', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({symbol})
    });
    fetchData();
}

fetchData();
setInterval(fetchData, 3000);
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print("===========================================================================")
    print(" 15m Supertrend [v5.0 触线即买版] 启动中... 监听端口: 8000")
    print("===========================================================================")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
