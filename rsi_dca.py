import pandas as pd
import numpy as np
import datetime
import time
from binance.client import Client


api_key = "your key"
secret_key = "your secret key"
client = Client(api_key=api_key, api_secret=secret_key, tld="com")
interval = Client.KLINE_INTERVAL_4HOUR
date = "7 days ago UTC"
rsi_threshold = 30
qfl_threshold = True
min_order_size = 10.1  # минимальный размер ордера на Бинансе
target_profit = 10  # профит позиции в % при котором позиция закроется маркетно
percentage_offset = 0.01 / 100  # отступ от текущей цены для выставления ордера
token = [
        "BTC",
        "ETH",
        "ADA",
        "BNB",
        "NEO",
        "LTC",
        "MINA"
]
base_token = "USDT"
price_lotsize = {
    "BTCUSDT": 2,
    "ETHUSDT": 2,
    "ADAUSDT": 4,
    "BNBUSDT": 1,
    "NEOUSDT": 2,
    "LTCUSDT": 2,
    "MINAUSDT": 3
}
amount_lotsize = {
    "BTCUSDT": 5,
    "ETHUSDT": 4,
    "ADAUSDT": 1,
    "BNBUSDT": 3,
    "NEOUSDT": 2,
    "LTCUSDT": 3,
    "MINAUSDT": 1
}
order_history = pd.DataFrame({
                    'date': [],
                    'traiding pair': [],
                    'amount': [],
                    'price': [],
                    'total': []
                    })


# Получаем балансы USDT аккаунта, доступные для выставления ордера
def get_balance_usdt():
    get_balance = client.get_asset_balance(asset='USDT')
    get_balance["free"] = pd.to_numeric(get_balance['free'], errors="coerce")
    return get_balance["free"]


# Получаем последнюю цену торговой пары
def get_last_price(trading_pair, interval, date):
    klines = client.get_historical_klines(trading_pair, interval, date)
    last_price = pd.DataFrame(klines)
    last_price.columns = [
        "open time", "open", "high", "low", "close",
        "volume", "close time", "quote asset volume",
        "number of trades", "taker buy base asset volume",
        "taker buy quote asset volume", "ignore"
        ]
    last_price["close"] = pd.to_numeric(last_price['close'], errors="coerce")
    last_price = last_price["close"].iat[-1]
    return last_price


# Получаем маркет дату
def get_ohlc(trading_pair, interval, date):
    klines = client.get_historical_klines(trading_pair, interval, date)
    ohlc = pd.DataFrame(klines)
    ohlc.columns = ["open time", "open", "high", "low", "close",
                    "volume", "close time", "quote asset volume",
                    "number of trades", "taker buy base asset volume",
                    "taker buy quote asset volume", "ignore"]
    ohlc["date"] = pd.to_datetime(ohlc["open time"], unit="ms")
    ohlc["close"] = pd.to_numeric(ohlc['close'], errors="coerce")
    ohlc["high"] = pd.to_numeric(ohlc['high'], errors="coerce")
    ohlc["low"] = pd.to_numeric(ohlc['low'], errors="coerce")
    ohlc["volume"] = pd.to_numeric(ohlc['volume'], errors="coerce")
    ohlc.set_index("date", inplace=True)
    return ohlc


# Определяем размер ордера в зависимости от размера баланса
def set_order_size():
    initial_balance = get_balance_usdt()
    if initial_balance > 10.1 and initial_balance < 100:
        return 10.1
    elif initial_balance > 101 and initial_balance < 300:
        return 15
    elif initial_balance > 301 and initial_balance < 1000:
        return 20
    elif initial_balance > 1001 and initial_balance < 10000:
        return 50
    else:
        return 90


# Считаем индикатор QFL
def qfl_single_tf(
        ohlc: pd.DataFrame,
        volume_ma: int = 6,
        percentage: float = 3.5,
        percentage_sell: float = 3.5,
        max_base_age: int = 0,
        allow_consecutive_signals: bool = True):
    # Считаем обьем скользящей средней
    ohlc["volume_ma"] = ohlc["volume"].rolling(volume_ma).mean()

    # Считаем значение QFL сигнала
    ohlc["down"] = (
        (ohlc["low"].shift(3) > ohlc["low"].shift(4))
        & (ohlc["low"].shift(4) > ohlc["low"].shift(5))
        & (ohlc["low"].shift(2) < ohlc["low"].shift(3))
        & (ohlc["low"].shift(1) < ohlc["low"].shift(2))
        & (ohlc["volume"].shift(3) > ohlc["volume_ma"].shift(3))
    )

    # Считаем значения fractal
    fractal_down = []
    for i in range(len(ohlc)):
        if ohlc.iloc[i]["up"]:
            fractal_up.append(ohlc.iloc[i]["high"])
        elif len(fractal_up) > 0:
            fractal_up.append(fractal_up[-1])
        else:
            fractal_up.append(None)

        if ohlc.iloc[i]["down"]:
            fractal_down.append(ohlc.iloc[i]["low"])
        elif len(fractal_down) > 0:
            fractal_down.append(fractal_down[-1])
        else:
            fractal_down.append(None)

    # Добавляем значения в датасэт
    ohlc["fractal_down"] = fractal_down

    # Считаем бары с момента последнего увеличения/уменьшения бара
    ohlc["age"] = ohlc["fractal_down"].notnull().cumsum()
    ohlc["age"] = ohlc.apply(
        lambda x: x["age"] if x["fractal_down"] is not None else None, axis=1
    )
    ohlc["age"] = ohlc["age"].fillna(method="ffill")

    # Считаем сигнали на покупку/продажу
    ohlc["buy"] = (ohlc["close"] / ohlc["fractal_down"]) < (1 - percentage / 100)

    # Фильтруем сигнал на покупку
    def shift_if_not_float(x):
        if isinstance(x, np.float):
            return x
        return x.shift(1)
    ohlc["buy"] = ohlc.apply(
        lambda x: x["buy"]
        and (
            allow_consecutive_signals
            or x["fractal_down"] != shift_if_not_float(x["fractal_down"])
        )
        and (max_base_age == 0 or x["age"] < max_base_age),
        axis=1,
    )
    return ohlc


# Рассчитываем индикатор RSI
def calculate_rsi(ohlc: pd.DataFrame, period: int = 14, round_rsi: bool = True):
    delta = ohlc["close"].diff()
    up = delta.copy()
    up[up < 0] = 0
    up = pd.Series.ewm(up, alpha=1/period).mean()
    down = delta.copy()
    down[down > 0] = 0
    down *= -1
    down = pd.Series.ewm(down, alpha=1/period).mean()
    ohlc["rsi"] = np.where(
        up == 0, 0, np.where(down == 0, 100, 100 - (100 / (1 + up / down)))
    )
    return ohlc


# Запускаем dca стратегию
def start_dca():
    while True:
        # Проверяем баланс base_token на соответствие минимального ордер сайза
        initial_balance = get_balance_usdt()
        if (initial_balance < min_order_size).all():
            print("Total order value should be more than 10 USDT")
        # Получаем маркет дату по каждой торговой паре и считаем индикаторы
        else:
            for i in range(0, len(token)):
                traiding_pair = (token[i] + base_token)
                ohlc = get_ohlc(traiding_pair, interval, date)
                rsi = calculate_rsi(ohlc)
                qfl = qfl_single_tf(ohlc)
                now = datetime.datetime.now()
                last_qfl = qfl["buy"].iat[-1]
                last_rsi = rsi["rsi"].iat[-1]
                # Проверяем индикаторы на соответветствие трешхолду
                if last_rsi < rsi_threshold and last_qfl == qfl_threshold:
                    token_price = ohlc["close"].iat[-1]
                    token_price_offset = np.round(
                        token_price * (percentage_offset), price_lotsize[traiding_pair]
                    )
                    order_price = np.round(token_price - token_price_offset, price_lotsize[traiding_pair])
                    order_amount = np.round(set_order_size() / order_price, amount_lotsize[traiding_pair])
                    order_total = np.round(order_price * order_amount, price_lotsize[traiding_pair])
                    # Выставляем лимитный ордер с небольшим отступом, чтобы исполнился
                    client.order_limit_buy(
                        symbol=traiding_pair, quantity=order_amount, price=str(order_price)
                    )
                    # Структурируем данные о сделке
                    order_info = pd.DataFrame({
                                    'date': [now],
                                    'traiding pair': [traiding_pair],
                                    'amount': [order_amount],
                                    'price': [order_price],
                                    'total': [order_total]
                                    })
                    # Добавляем данные о сделке в датасэт
                    global order_history
                    order_history = order_history.append(order_info, ignore_index=True)
                    # Считаем позицию и пнл по текущей паре
                    position = (
                        order_history.groupby("traiding pair")
                        .apply(
                            lambda x: x[["amount", "total"]][x["traiding pair"] == traiding_pair].cumsum()
                        )
                        .reset_index(level=0, drop=True)
                    )
                    position['traiding pair'] = traiding_pair
                    position['unrealized_pnl'] = 1 - position['total'] / (position['amount'] * token_price)
                    position['close_trigger'] = position['unrealized_pnl'] >= target_profit
                    # считаем ордер для закрытия позиции
                    order_sell = str(np.round(position['amount'].iat[-1], amount_lotsize[traiding_pair]))
                    close_position_trigger = position['close_trigger'].iat[-1]
                    # проверяем pnl на тригер и закрываем позицию по торговой паре
                    if close_position_trigger is True:
                        client.order_market_sell(symbol=traiding_pair, quantity=order_sell)
                        # обнуляем значения истории торгов по торговой паре
                        order_history = order_history[order_history['traiding pair'] != traiding_pair]
                    print(
                        f"Placed limit buy order for {order_amount:.8f} {token[i]} at a price of {order_price:.4f} USDT"
                    )
                    break
                else:
                    print(
                        f" {now} \n {traiding_pair} QFL signal and RSI indicator does not meet require conditions \n QFL:{last_qfl} \n RSI:{last_rsi}"
                    )
        time.sleep(3600)
