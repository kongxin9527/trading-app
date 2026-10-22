import asyncio
import json
import time
import datetime
from collections import deque
import aiohttp
import websockets
import pandas as pd
from dotenv import load_dotenv
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 加载 .env 环境变量
load_dotenv()
TG_BOT_TOKEN = os.getenv('TG_BOT_TOKEN')
TG_CHAT_ID = os.getenv('TG_CHAT_ID')
MIN_24H_VOLUME = float(os.getenv('MIN_24H_VOLUME', 5000000))
PUMP_THRESHOLD_1 = float(os.getenv('PUMP_THRESHOLD_1', 0.08))
PUMP_THRESHOLD_2 = float(os.getenv('PUMP_THRESHOLD_2', 0.15))
COOLDOWN_MINUTES = int(os.getenv('COOLDOWN_MINUTES', 60))
VOLUME_MULTIPLIER = float(os.getenv('VOLUME_MULTIPLIER', 3.0))

# 内存状态管理
active_symbols = set()          # 当前满足 500万 交易额的币种
price_windows = {}              # 记录币种过去15分钟的价格队列 {symbol: deque([(timestamp, price)])}
cooldown_cache = {}             # 冷却池 {symbol: {"expire": timestamp, "base_price": float, "hit_level_2": bool}}
daily_triggers = {}             # 每日触发次数 {symbol: count}
last_daily_report = None        # 记录心跳日报发送时间

# Supertrend 计算 (15m周期，标准参数 10, 3)
def calculate_supertrend(df, period=10, multiplier=3):
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift(1))
    df['tr3'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=period).mean()
    
    hl2 = (df['high'] + df['low']) / 2
    df['basic_ub'] = hl2 + (multiplier * df['atr'])
    df['basic_lb'] = hl2 - (multiplier * df['atr'])
    
    df['final_ub'] = df['basic_ub']
    df['final_lb'] = df['basic_lb']
    df['trend'] = 1 # 1 for bullish, -1 for bearish
    
    for i in range(period, len(df)):
        # Calculate Final Upper Band
        if df['basic_ub'].iloc[i] < df['final_ub'].iloc[i-1] or df['close'].iloc[i-1] > df['final_ub'].iloc[i-1]:
            df.loc[df.index[i], 'final_ub'] = df['basic_ub'].iloc[i]
        else:
            df.loc[df.index[i], 'final_ub'] = df['final_ub'].iloc[i-1]
            
        # Calculate Final Lower Band
        if df['basic_lb'].iloc[i] > df['final_lb'].iloc[i-1] or df['close'].iloc[i-1] < df['final_lb'].iloc[i-1]:
            df.loc[df.index[i], 'final_lb'] = df['basic_lb'].iloc[i]
        else:
            df.loc[df.index[i], 'final_lb'] = df['final_lb'].iloc[i-1]
            
        # Calculate Trend
        if df['close'].iloc[i] > df['final_ub'].iloc[i-1]:
            df.loc[df.index[i], 'trend'] = 1
        elif df['close'].iloc[i] < df['final_lb'].iloc[i-1]:
            df.loc[df.index[i], 'trend'] = -1
        else:
            df.loc[df.index[i], 'trend'] = df['trend'].iloc[i-1]
            
    return df['trend'].iloc[-1] == 1 # 返回当前是否是多头 (1)

async def send_telegram_msg(session, text):
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TG_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    try:
        await session.post(url, json=payload)
    except Exception as e:
        logging.error(f"TG发送失败: {e}")

async def verify_and_alert(session, symbol, current_price, lowest_price, vol_24h, is_level_2=False):
    """触发后，拉取 REST 数据进行二次核验（量比、Supertrend、费率）"""
    global cooldown_cache, daily_triggers
    
    try:
        # 1. 获取最近的 15m K线（过去4小时需拉取 16 根K线，为了计算均线和Supertrend，多拉一点至 40 根）
        url_klines = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval=15m&limit=40"
        async with session.get(url_klines) as resp:
            data = await resp.json()
            
        if len(data) < 20: return
        
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        df = df.astype(float)
        
        # 验证 1: Supertrend 是否多头
        is_bullish = calculate_supertrend(df)
        if not is_bullish:
            logging.info(f"[{symbol}] 放弃: 15m Supertrend 非多头状态")
            return
            
        # 验证 2: 15m量比验证 (当前15分钟交易量 vs 过去4小时平均每15分钟交易量)
        # 取过去 16 根K线 (4小时) 的平均量（不含当前这根未收线的）
        past_4h_vol_avg = df['quote_asset_volume'].iloc[-17:-1].mean()
        current_15m_vol = df['quote_asset_volume'].iloc[-1]
        
        vol_multiplier = current_15m_vol / past_4h_vol_avg if past_4h_vol_avg > 0 else 0
        if vol_multiplier < VOLUME_MULTIPLIER:
            logging.info(f"[{symbol}] 放弃: 量能放大不足 ({vol_multiplier:.1f}x < {VOLUME_MULTIPLIER}x)")
            return

        # 验证 3: 获取实时资金费率
        url_premium = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={symbol}"
        funding_rate = 0.0
        async with session.get(url_premium) as resp:
            fr_data = await resp.json()
            funding_rate = float(fr_data.get('lastFundingRate', 0))

        # --- 校验全部通过，准备发送电报 ---
        
        # 统计今日触发次数
        today_date = datetime.datetime.utcnow().date()
        if symbol not in daily_triggers or daily_triggers[symbol]['date'] != today_date:
            daily_triggers[symbol] = {'date': today_date, 'count': 0}
        daily_triggers[symbol]['count'] += 1
        d_count = daily_triggers[symbol]['count']
        
        # 更新冷却缓存
        cooldown_cache[symbol] = {
            "expire": time.time() + (COOLDOWN_MINUTES * 60),
            "base_price": lowest_price,
            "hit_level_2": is_level_2
        }

        # 计算参考位
        defense_price = current_price * 0.96  # -4%
        target_price = current_price * 1.12   # +12%
        rise_pct = ((current_price - lowest_price) / lowest_price) * 100
        
        # 拼装消息面板
        header_icon = "🔥 【二次加速报警】" if is_level_2 else "🚀 【山寨异动拉升】"
        msg = f"""{header_icon} <b>#{symbol}</b> (今日第 {d_count} 次)
━━━━━━━━━━━━━━━
📈 <b>15m 区间涨幅</b>：+{rise_pct:.2f}% (起爆: {lowest_price} -> 现价: {current_price})
📊 <b>量能爆发倍数</b>：{vol_multiplier:.1f}x (15m成交额: ${current_15m_vol/10000:.0f}万)
🌊 <b>24h 总体概况</b>：成交额 ${vol_24h/100000000:.2f}亿
💰 <b>资金费率</b>：{funding_rate * 100:.4f}%
━━━━━━━━━━━━━━━
🎯 <b>快捷参考位 (按现价估算)</b>：
• 建议防守位 (-4%)：{defense_price:.4f}
• 短期目标位 (+12%)：{target_price:.4f}
⏱ <b>触发时间</b>：{datetime.datetime.now().strftime('%H:%M:%S')}
🔗 <a href="https://www.binance.com/zh-CN/futures/{symbol.replace('USDT','')}USDT">点击直达币安交易页面</a>"""

        await send_telegram_msg(session, msg)
        logging.info(f"[{symbol}] 报警已发送！涨幅: {rise_pct:.2f}%")

    except Exception as e:
        logging.error(f"核验 {symbol} 数据时出错: {e}")

async def process_ticker(session, ticker_data):
    """处理每一秒的 Ticker 数据，计算滑动窗口"""
    global active_symbols, price_windows, cooldown_cache
    
    symbol = ticker_data['s']
    if not symbol.endswith('USDT') or symbol in ['USDCUSDT', 'BUSDUSDT']:
        return # 过滤稳定币和非USDT交易对
        
    vol_24h = float(ticker_data['q']) # 24小时成交额 (USDT)
    current_price = float(ticker_data['c'])
    now = time.time()
    
    # 动态标的池维护
    if vol_24h >= MIN_24H_VOLUME:
        if symbol not in active_symbols:
            active_symbols.add(symbol)
            price_windows[symbol] = deque(maxlen=900) # 15分钟，每秒最多一条
    else:
        if symbol in active_symbols:
            active_symbols.remove(symbol)
            if symbol in price_windows: del price_windows[symbol]
        return

    # 清理 15 分钟前的数据 (900秒)
    window = price_windows[symbol]
    window.append((now, current_price))
    while window and window[0][0] < now - 900:
        window.popleft()
        
    # 如果窗口数据建立不够（不到 5 分钟），暂不监控，防止初始瞬间假触发
    if window[-1][0] - window[0][0] < 300: 
        return

    # 寻找过去15分钟最低价
    lowest_price = min(item[1] for item in window)
    
    # 检查涨幅条件
    if lowest_price > 0:
        rise_ratio = (current_price - lowest_price) / lowest_price
        
        # 是否在冷却期中
        if symbol in cooldown_cache and cooldown_cache[symbol]['expire'] > now:
            cache = cooldown_cache[symbol]
            # 检查二次加速 (基于起爆价，涨幅达到 15% 且未报过二次)
            second_rise_ratio = (current_price - cache['base_price']) / cache['base_price']
            if second_rise_ratio >= PUMP_THRESHOLD_2 and not cache['hit_level_2']:
                # 触发二次加速
                asyncio.create_task(verify_and_alert(session, symbol, current_price, cache['base_price'], vol_24h, is_level_2=True))
        else:
            # 正常初次触发 (涨幅 >= 8%)
            if rise_ratio >= PUMP_THRESHOLD_1:
                # 为了防止并发瞬间重复报警，先写入占位冷却时间
                cooldown_cache[symbol] = {"expire": now + 60, "base_price": lowest_price, "hit_level_2": False}
                asyncio.create_task(verify_and_alert(session, symbol, current_price, lowest_price, vol_24h, is_level_2=False))

async def heartbeat_task(session):
    """每天早上 8:00 发送一次心跳"""
    global last_daily_report
    while True:
        now = datetime.datetime.now()
        # 每天早上 08:00 (本地服务器时间) 发送
        if now.hour == 8 and now.minute == 0:
            today_str = now.strftime('%Y-%m-%d')
            if last_daily_report != today_str:
                active_count = len(active_symbols)
                msg = f"🤖 <b>扫描器运行正常</b>\n当前监控中活跃合约: {active_count} 个 (24H成交额>500万)"
                await send_telegram_msg(session, msg)
                last_daily_report = today_str
        await asyncio.sleep(60) # 每分钟检查一次

async def main_ws_loop():
    url = "wss://fstream.binance.com/ws/!ticker@arr" # U本位全市场实时 Ticker 流
    
    async with aiohttp.ClientSession() as session:
        # 启动心跳任务
        asyncio.create_task(heartbeat_task(session))
        
        while True:
            try:
                logging.info("正在连接 Binance WebSocket...")
                async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                    logging.info("连接成功，开始实时监控...")
                    while True:
                        res = await ws.recv()
                        data = json.loads(res)
                        for ticker in data:
                            await process_ticker(session, ticker)
            except Exception as e:
                logging.error(f"WebSocket 断开或错误: {e}，5秒后自动重连...")
                await asyncio.sleep(5)

if __name__ == "__main__":
    logging.info("==== 币安山寨异动专业监控系统启动 ====")
    asyncio.run(main_ws_loop())
