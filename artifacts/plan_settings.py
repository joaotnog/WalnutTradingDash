from artifacts.mapping import *
from datetime import datetime as dt


free_plan_list = dict(cryptos = list(map_code2crypto.keys())[::2],
                      technicals = list(map_tech2fun.keys())[::2],
                      date_min = dt(2021,6,1),
                      date_max = dt(2022,1,1))
