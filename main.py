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

# 内存状态管理（终极升级版）
price_windows = {}              # 无差别记录全市场过去15分钟轨迹 {symbol: deque([(timestamp, price)])}
current_volumes = {}            # 实时记录全市场 24H 成交额
cooldown_cache = {}             # 冷却池
daily_triggers = {}             
last_daily_report = None        

# Supertrend 计算
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
    df['trend'] = 1 
    
    for i in range(period, len(df)):
        if df['basic_ub'].iloc[i] < df['final_ub'].iloc[i-1] or df['close'].iloc[i-1] > df['final_ub'].iloc[i-1]:
            df.loc[df.index[i], 'final_ub'] = df['basic_ub'].iloc[i]
        else:
            df.loc[df.index[i], 'final_ub'] = df['final_ub'].iloc[i-1]
            
        if df['basic_lb'].iloc[i] > df['final_lb'].iloc[i-1] or df['close'].iloc[i-1] < df['final_lb'].iloc[i-1]:
            df.loc[df.index[i], 'final_lb'] = df['basic_lb'].iloc[i]
        else:
            df.loc[df.index[i], 'final_lb'] = df['final_lb'].iloc[i-1]
            
        if df['close'].iloc[i] > df['final_ub'].iloc[i-1]:
            df.loc[df.index[i], 'trend'] = 1
        elif df['close'].iloc[i] < df['final_lb'].iloc[i-1]:
            df.loc[df.index[i], 'trend'] = -1
        else:
            df.loc[df.index[i], 'trend'] = df['trend'].iloc[i-1]
            
    return df['trend'].iloc[-1] == 1 

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

async def fetch_historical_klines(session, symbol, sem):
    """并发拉取历史K线，用于开机预热"""
    async with sem:
        try:
            url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval=1m&limit=15"
            async with session.get(url) as resp:
                data = await resp.json()
                window = deque(maxlen=900)
                for k in data:
                    ts = k[0] / 1000.0  # 转为秒
                    low_price = float(k[3]) # 用 1m K线的最低价作为轨迹点
                    window.append((ts, low_price))
                price_windows[symbol] = window
        except Exception:
            pass

async def pre_heat(session):
    """【新增】冷启动数据预热：瞬间补齐开机前的 15 分钟历史数据"""
    logging.info("开始冷启动数据预热...")
    url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
    async with session.get(url) as resp:
        tickers = await resp.json()
        
    tasks = []
    sem = asyncio.Semaphore(20) # 限制并发数，防止触发币安限流
    for t in tickers:
        symbol = t['symbol']
        if not symbol.endswith('USDT') or symbol in ['USDCUSDT', 'BUSDUSDT']:
            continue
        current_volumes[symbol] = float(t['quoteVolume'])
        # 全市场 U本位合约历史补齐
        tasks.append(fetch_historical_klines(session, symbol, sem))
        
    await asyncio.gather(*tasks)
    logging.info(f"数据预热完成！共预装 {len(price_windows)} 个合约。")

async def verify_and_alert(session, symbol, current_price, lowest_price, vol_24h, is_level_2=False):
    """深度核验与报警"""
    global cooldown_cache, daily_triggers
    try:
        url_klines = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval=15m&limit=40"
        async with session.get(url_klines) as resp:
            data = await resp.json()
            
        if len(data) < 20: return
        
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        df = df.astype(float)
        
        # 1. 验证 Supertrend
        if not calculate_supertrend(df):
            logging.info(f"[{symbol}] 放弃: 15m Supertrend 非多头")
            return
            
        # 2. 验证量比
        past_4h_vol_avg = df['quote_asset_volume'].iloc[-17:-1].mean()
        current_15m_vol = df['quote_asset_volume'].iloc[-1]
        
        vol_multiplier = current_15m_vol / past_4h_vol_avg if past_4h_vol_avg > 0 else 0
        if vol_multiplier < VOLUME_MULTIPLIER:
            logging.info(f"[{symbol}] 放弃: 量能放大不足 ({vol_multiplier:.1f}x)")
            return

        # 3. 获取资金费率
        url_premium = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={symbol}"
        funding_rate = 0.0
        async with session.get(url_premium) as resp:
            fr_data = await resp.json()
            funding_rate = float(fr_data.get('lastFundingRate', 0))

        today_date = datetime.datetime.utcnow().date()
        if symbol not in daily_triggers or daily_triggers[symbol]['date'] != today_date:
            daily_triggers[symbol] = {'date': today_date, 'count': 0}
        daily_triggers[symbol]['count'] += 1
        d_count = daily_triggers[symbol]['count']
        
        cooldown_cache[symbol] = {
            "expire": time.time() + (COOLDOWN_MINUTES * 60),
            "base_price": lowest_price,
            "hit_level_2": is_level_2
        }

        defense_price = current_price * 0.96 
        target_price = current_price * 1.12  
        rise_pct = ((current_price - lowest_price) / lowest_price) * 100
        
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
        logging.error(f"核验 {symbol} 出错: {e}")

async def process_ticker(session, ticker_data):
    """处理 WebSocket 推送"""
    global price_windows, current_volumes, cooldown_cache
    
    symbol = ticker_data['s']
    if not symbol.endswith('USDT') or symbol in ['USDCUSDT', 'BUSDUSDT']:
        return 
        
    vol_24h = float(ticker_data['q']) 
    current_price = float(ticker_data['c'])
    now = time.time()
    
    # 实时更新该币种成交额
    current_volumes[symbol] = vol_24h
    
    # 【逻辑升级】无差别记录价格轨迹，彻底解决300万->500万突破盲区
    if symbol not in price_windows:
        price_windows[symbol] = deque(maxlen=900) 
        
    window = price_windows[symbol]
    window.append((now, current_price))
    
    while window and window[0][0] < now - 900:
        window.popleft()
        
    # 【成交额风控】小于 500 万，只记录轨迹，不触发报警！
    if vol_24h < MIN_24H_VOLUME:
        return

    # 数据收集跨度不足 5 分钟不处理（预热后的数据会自动跳过此限制）
    if window[-1][0] - window[0][0] < 300: 
        return

    lowest_price = min(item[1] for item in window)
    
    if lowest_price > 0:
        rise_ratio = (current_price - lowest_price) / lowest_price
        
        if symbol in cooldown_cache and cooldown_cache[symbol]['expire'] > now:
            cache = cooldown_cache[symbol]
            second_rise_ratio = (current_price - cache['base_price']) / cache['base_price']
            if second_rise_ratio >= PUMP_THRESHOLD_2 and not cache['hit_level_2']:
                asyncio.create_task(verify_and_alert(session, symbol, current_price, cache['base_price'], vol_24h, is_level_2=True))
        else:
            if rise_ratio >= PUMP_THRESHOLD_1:
                cooldown_cache[symbol] = {"expire": now + 60, "base_price": lowest_price, "hit_level_2": False}
                asyncio.create_task(verify_and_alert(session, symbol, current_price, lowest_price, vol_24h, is_level_2=False))

async def heartbeat_task(session):
    """每日早报"""
    global last_daily_report
    while True:
        now = datetime.datetime.now()
        # 每天早上 08:00 发送
        if now.hour == 8 and now.minute == 0:
            today_str = now.strftime('%Y-%m-%d')
            if last_daily_report != today_str:
                active_count = sum(1 for v in current_volumes.values() if v >= MIN_24H_VOLUME)
                msg = f"🤖 <b>扫描器每日简报</b>\n系统运行正常，当前监控中达标活跃合约: {active_count} 个 (24H成交额>500万)"
                await send_telegram_msg(session, msg)
                last_daily_report = today_str
        await asyncio.sleep(30) # 每 30 秒检查，防止错过整点

async def main_ws_loop():
    # 绝对正确的最新版常规市场数据流地址
    url = "wss://fstream.binance.com/market/ws/!ticker@arr" 
    
    async with aiohttp.ClientSession() as session:
        # 1. 启动先预热数据
        await pre_heat(session)
        
        # 2. 发送开机问候
        active_count = sum(1 for v in current_volumes.values() if v >= MIN_24H_VOLUME)
        await send_telegram_msg(session, f"🚀 <b>山寨雷达启动成功</b>\n历史数据预热完毕，当前监控中达标合约: {active_count} 个。")
        
        # 3. 挂载日常心跳
        asyncio.create_task(heartbeat_task(session))
        
        # 定义用于测试接收状态的计数器
        msg_count = 0
        last_count_time = time.time()
        
        while True:
            try:
                logging.info(f"正在连接 Binance WebSocket 网关: {url} ...")
                # 直接连接这个特定的市场数据流，连上就会源源不断下发数据
                async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                    logging.info("网关接入成功，开始接收实时数据...")
                    
                    while True:
                        res = await ws.recv()
                        data = json.loads(res)
                        
                        # -----------------------
                        # 接收状态日志打印
                        msg_count += 1
                        now = time.time()
                        if now - last_count_time >= 60:
                            logging.info(f"📊 WS流状态正常: 过去 1 分钟共接收 {msg_count} 次数据推送")
                            msg_count = 0
                            last_count_time = now
                        # -----------------------
                            
                        # 如果接收到的是数据列表，开始交给策略处理
                        if isinstance(data, list):
                            for ticker in data:
                                await process_ticker(session, ticker)
                                
            except Exception as e:
                logging.error(f"WebSocket 断线或无响应: {e}，5秒后自动重连...")
                await asyncio.sleep(5)
                
if __name__ == "__main__":
    logging.info("==== 币安山寨异动专业监控系统启动 ====")
    asyncio.run(main_ws_loop())
