import ccxt
import pandas as pd
import pandas_ta as ta
import time
import datetime
import requests
import os
from dotenv import load_dotenv

# ==========================================
# 1. 基础配置与 Telegram 接入
# ==========================================
load_dotenv() # 加载 .env 文件
TG_TOKEN = os.getenv("TG_TOKEN")
TG_CHAT_ID = os.getenv("TG_CHAT_ID")

def send_telegram(message):
    if not TG_TOKEN or not TG_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": TG_CHAT_ID, "text": message, "parse_mode": "Markdown"}, timeout=5)
    except Exception as e:
        print(f"TG推送失败: {e}")

exchange = ccxt.binance({
    'enableRateLimit': True,
    'options': {'defaultType': 'future'}
})

# ==========================================
# 2. 状态机与时间控制变量
# ==========================================
screening_pool = set()   # 海选池：存放 24H 涨幅曾 > 15% 的币
core_watchlist = {}      # 核心白名单：{'PEPE/USDT:USDT': {'ema90_1h': 0.1, 'ema90_4h': 0.08, 'last_alert': 0}}

# 时间锚点
POOL_RESET_INTERVAL = 4 * 24 * 60 * 60  # 4天 (秒)
LOW_FREQ_INTERVAL = 15 * 60             # 15分钟 (秒)
COOLDOWN_TIME = 2 * 60 * 60             # 报警冷却 2 小时 (秒)
HIGH_FREQ_INTERVAL = 5                  # 高频轮询 5 秒

last_pool_reset = time.time()
last_low_freq_check = 0
last_hourly_update = -1  # 记录上一次整点更新的小时数

print("🚀 工业级 EMA90 动能狙击系统启动...")
send_telegram("🟢 **量化系统已启动**\n策略: 24h高热度 -> EMA90动能锁定 -> 伏击回踩")

# ==========================================
# 3. 辅助函数：计算均线
# ==========================================
def get_emas(symbol, timeframe, limit=150):
    """获取指定周期的 K 线并计算 EMA90"""
    try:
        bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if len(bars) < 100: return None
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['EMA20'] = ta.ema(df['close'], length=20)
        df['EMA90'] = ta.ema(df['close'], length=90)
        return df
    except Exception as e:
        print(f"获取 {symbol} {timeframe} 数据失败: {e}")
        return None

# ==========================================
# 4. 主循环架构
# ==========================================
while True:
    current_time = time.time()
    now_dt = datetime.datetime.now()
    
    # -----------------------------------------------------------
    # 【机制 A】4天全局重置清洗 (保留核心池)
    # -----------------------------------------------------------
    if current_time - last_pool_reset > POOL_RESET_INTERVAL:
        old_pool_size = len(screening_pool)
        screening_pool = set(core_watchlist.keys()) # 清空海选池，但把核心白名单的币加回来
        last_pool_reset = current_time
        print(f"\n[{now_dt.strftime('%H:%M:%S')}] 🧹 执行4天大清洗！旧池规模: {old_pool_size}，清洗后: {len(screening_pool)}")
        send_telegram(f"🧹 **内存池已重置 (4天周期)**\n保留核心监控目标数: {len(screening_pool)}")

    try:
        # -----------------------------------------------------------
        # 【机制 B】高频轮询 (每 5 秒，拉取全市场现价)
        # -----------------------------------------------------------
        live_tickers = exchange.fetch_tickers()
        
        # 1. 抓取海选池：24小时涨幅 > 15%
        for sym, data in live_tickers.items():
            if ':USDT' in sym:
                # ccxt 里的 percentage 是 24h 涨跌幅百分比
                pct_change = data.get('percentage', 0)
                if pct_change and pct_change > 15 and sym not in screening_pool:
                    screening_pool.add(sym)
                    print(f"  [入池] {sym} 24h涨幅突破15% ({pct_change}%)，进入海选池。")

        # 2. 核心白名单高频狙击
        for sym, target_data in core_watchlist.items():
            if sym in live_tickers:
                current_price = live_tickers[sym]['last']
                buy_target = target_data['ema90_1h'] * 1.01
                
                # 触及 1H EMA90 上方 1%
                if current_price <= buy_target:
                    # 检查 2小时 冷却期
                    if current_time - target_data['last_alert'] > COOLDOWN_TIME:
                        msg = (
                            f"💥 **买入警报触发** 💥\n"
                            f"🪙 币种: `{sym}`\n"
                            f"📉 现价触及: `{current_price:.5f}`\n"
                            f"🎯 EMA90伏击圈: `{buy_target:.5f}`\n"
                            f"⏳ (该币种警报将进入2小时冷却)"
                        )
                        send_telegram(msg)
                        print(f"[{now_dt.strftime('%H:%M:%S')}] 💥 触发买入：{sym} 现价 {current_price}")
                        target_data['last_alert'] = current_time # 刷新最后报警时间

        # -----------------------------------------------------------
        # 【机制 C】整点缓存更新与趋势防守 (每小时执行1次)
        # -----------------------------------------------------------
        if now_dt.hour != last_hourly_update:
            print(f"\n[{now_dt.strftime('%H:%M:%S')}] ⏰ 开始整点缓存更新...")
            symbols_to_remove = []
            
            for sym in list(core_watchlist.keys()):
                # 拉取 1H 和 4H 数据
                df_1h = get_emas(sym, '1h', limit=100)
                df_4h = get_emas(sym, '4h', limit=100)
                
                if df_1h is not None and df_4h is not None:
                    latest_1h_close = df_1h.iloc[-1]['close']
                    ema90_1h_new = df_1h.iloc[-1]['EMA90']
                    ema90_4h_new = df_4h.iloc[-1]['EMA90']
                    
                    # 趋势防守：如果 1H 的最新收盘价 跌破了 4H EMA90，视为破位
                    if latest_1h_close < ema90_4h_new:
                        symbols_to_remove.append(sym)
                        msg = f"💔 **趋势破位剔除**\n🪙 `{sym}` 收盘跌破 4H EMA90，已移出核心白名单。"
                        send_telegram(msg)
                        print(f"  [出局] {sym} 收盘价 {latest_1h_close} 跌破 4H EMA90 {ema90_4h_new}")
                    else:
                        # 正常更新缓存
                        core_watchlist[sym]['ema90_1h'] = ema90_1h_new
                        core_watchlist[sym]['ema90_4h'] = ema90_4h_new
                        
                time.sleep(0.1) # 限速
                
            # 执行删除
            for sym in symbols_to_remove:
                del core_watchlist[sym]
                
            last_hourly_update = now_dt.hour
            print(f"⏰ 整点更新完成，核心白名单剩余 {len(core_watchlist)} 个币种。")

        # -----------------------------------------------------------
        # 【机制 D】低频动能验票入场 (每15分钟检查海选池)
        # -----------------------------------------------------------
        if current_time - last_low_freq_check > LOW_FREQ_INTERVAL:
            print(f"\n[{now_dt.strftime('%H:%M:%S')}] 🔍 执行15分钟动能验票... (海选池规模: {len(screening_pool)})")
            
            # 只检查在海选池，但还未进入核心池的币
            pending_symbols = [s for s in screening_pool if s not in core_watchlist]
            
            for sym in pending_symbols:
                df_1h = get_emas(sym, '1h', limit=150)
                if df_1h is not None:
                    # 计算 72H 最高价
                    df_1h['High_72h'] = df_1h['high'].rolling(72).max()
                    last_bar = df_1h.iloc[-1]
                    
                    ema20 = last_bar['EMA20']
                    ema90 = last_bar['EMA90']
                    highest = last_bar['High_72h']
                    
                    if pd.notna(ema90) and pd.notna(highest):
                        # 验票规则：多头排列 且 偏离度 > 15%
                        if ema20 > ema90 and ((highest - ema90) / ema90 > 0.15):
                            # 拉取 4H 均线初始化
                            df_4h = get_emas(sym, '4h', limit=100)
                            ema90_4h = df_4h.iloc[-1]['EMA90'] if df_4h is not None else ema90 * 0.9 # 容错
                            
                            # 门票生效，永久锁定进入核心白名单
                            core_watchlist[sym] = {
                                'ema90_1h': ema90,
                                'ema90_4h': ema90_4h,
                                'last_alert': 0 # 初始报警冷却为0
                            }
                            
                            msg = f"🎯 **晋级核心狙击池**\n🪙 `{sym}` 验票通过(动能爆发)\n💰 当前 1H EMA90: `{ema90:.5f}`\n🛡 4H 趋势防守线: `{ema90_4h:.5f}`"
                            send_telegram(msg)
                            print(f"  [晋级锁定] {sym} 动能条件达成，已加入核心狙击池！")
                
                time.sleep(0.05) # 防封锁限速
                
            last_low_freq_check = current_time
            print(f"🔍 验票结束，核心池当前监控: {list(core_watchlist.keys())}")

    except ccxt.NetworkError as e:
        print(f"网络抖动，稍后重试: {e}")
    except Exception as e:
        print(f"系统运行中出现异常: {e}")

    # 高频心跳休眠
    time.sleep(HIGH_FREQ_INTERVAL)
