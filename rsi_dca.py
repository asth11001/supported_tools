import pandas as pd
import numpy as np
import time
from binance.client import Client


api_key = "your key"
secret_key = "your secret key"
client = Client(api_key = api_key, api_secret = secret_key, tld = "com")
trading_pair = "BTCUSDT"
interval = Client.KLINE_INTERVAL_4HOUR
date = "1 Jan, 2017" 
rsi_threshold = 30 
min_order_size = 10.1 #минимальный размер ордера на Бинансе


def get_balance_usdt ():
    #получаем балансы USDT аккаунта, доступные для выставления ордера
    get_balance = client.get_asset_balance(asset = 'USDT')
    get_balance["free"] = pd.to_numeric(get_balance['free'], errors = "coerce")
    return get_balance["free"]


def get_ohlc(trading_pair, interval, date):
    #получаем маркет дату
    klines = client.get_historical_klines(trading_pair, interval, date)
    ohlc = pd.DataFrame(klines)
    ohlc.columns = ["Open Time", "Open", "High", "Low", "Close",
                    "Volume", "Close Time", "Quote Asset Volume", 
                    "Number of Trades", "Taker Buy Base Asset Volume",
                    "Taker Buy Quote Asset Volume", "Ignore"]
    ohlc["Date"] = pd.to_datetime(ohlc["Open Time"], unit = "ms")
    ohlc["Close"] = pd.to_numeric(ohlc['Close'], errors = "coerce")
    ohlc.set_index("Date", inplace = True)
    
    return ohlc


def set_order_size():
    #определяем размер ордера в зависимости от размера баланса
    initial_balance = get_balance_usdt()
    if ((initial_balance > 10.1) & (initial_balance < 100)).all():
        order_size = 10.1
        return order_size
    elif ((initial_balance > 101) & (initial_balance < 300)).all():
        order_size = 15
        return order_size
    elif ((initial_balance > 301) & (initial_balance < 1000)).all():
        order_size = 20
        return order_size
    elif ((initial_balance > 1001) & (initial_balance < 10000)).all():
        order_size = 50
        return order_size
    else:
        order_size = 90
        return order_size
    
    
def calculate_rsi(ohlc: pd.DataFrame, period: int = 14, round_rsi: bool = True):
    #рассчитывае индикатор RSI
    delta = ohlc["Close"].diff()
    up = delta.copy()
    up[up < 0] = 0
    up = pd.Series.ewm(up, alpha=1/period).mean()
    down = delta.copy()
    down[down > 0] = 0
    down *= -1
    down = pd.Series.ewm(down, alpha=1/period).mean()
    rsi = np.where(up == 0, 0, np.where(down == 0, 100, 100 - (100 / (1 + up / down))))
    
    return np.round(rsi, 2) if round_rsi else rsi


    
def start_dca():    
    #запускаем dca стратегию
    while True:   
        ohlc = get_ohlc(trading_pair, interval, date)
        all_rsi = calculate_rsi(ohlc)
        df = pd.DataFrame(all_rsi)
        last_rsi = df.iloc[ -1]
        initial_balance = get_balance_usdt() 
        #проверяем на соответствие требования минимального размера ордера
        if (initial_balance < min_order_size).all():
            print ("Total order value should be more than 10 USDT")
        elif (last_rsi < rsi_threshold).all():
            # рассчитываем размер ордера в BTC
            btc_price = ohlc["Close"].iat[-1]
            btc_amount = np.round(set_order_size()  / btc_price, 4)
            # Выставляем ордер маркетный/лимитный
            order = client.order_market_buy(
                symbol=trading_pair,
                quantity=btc_amount
            )

            print(f'Placed market buy order for {btc_amount:.8f} BTC at a price of {btc_price:.2f} USDT')
        else:
            print (f' The RSI indicator does not meet require conditions \n {last_rsi}')
        time.sleep(14400)
        
        
start_dca()
