from artifacts.mapping import *
from datetime import datetime as dt

tech_available = ['MOM (Momentum Indicator)',
              # 'SMA (Simple Moving Average)',
              'STOCH (Stochastic Oscillator)',
              'EMA (Exponential Moving Average)',
              'CMO (Chande Momentum Oscillator)',
              'ADX (Average Directional Movement Index)',
              'ATR (Average True Range)',
              'AROON (Aroon Oscillator)',
              # 'PSAR (Parabolic Stop and Reverse)',
              'RSI (Relative Strength Index)',
              'MACD (Moving Average Convergence Divergence)',
              # 'BBANDS (Bollinger Bands)',
              'TRIX (Triple Exponential)',
              'MFI (Money Flow Index)']
tech_available.sort()
tech_available = ['SMA (Simple Moving Average)'] + tech_available  

free_plan_list = dict(cryptos = list(map_code2crypto.keys())[::2],
                      technicals = tech_available[::2],
                      date_min = dt(2021,6,1),
                      date_max = dt(2022,1,1))
