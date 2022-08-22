import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime as dt
import ffn
from Functions import *
import pandas_ta as ta
from PIL import Image
import performanceanalytics.statistics as pas
import performanceanalytics.table.table as pat
import investpy as inv
import matplotlib.pyplot as plt
from artifacts.mapping import *
from artifacts.plan_settings import *
from datetime import timedelta
import sys
from utils import *
import altair as alt

# sys.argv = ['','FREE']
logo = Image.open('walnuttradingdash_logo2.png')

st.set_page_config(
    page_title = 'Backtest cryptocurrency trading strategies',
    page_icon = logo,
    layout = 'wide'
)

st.sidebar.title('WalnutTradingDash')
st.sidebar.markdown('Backtest cryptocurrency trading strategies')
st.sidebar.markdown('')

if sys.argv[1]=='PRO':
    free_plan = False
if sys.argv[1]=='FREE':
    free_plan = True    

scripts = list(map_crypto2code.keys())
indicators = import_indicators()

backtest_timeframe = st.sidebar.expander('Time interval')

if free_plan: 
    start_date = backtest_timeframe.date_input('Starting Date', value = free_plan_list['date_min'], min_value = free_plan_list['date_min'], max_value = free_plan_list['date_max'])
    end_date = backtest_timeframe.date_input('Ending Date', value = free_plan_list['date_max'], min_value = free_plan_list['date_min'], max_value = free_plan_list['date_max'])
else:
    start_date = backtest_timeframe.date_input('Starting Date', value = dt(2017,1,1), min_value = dt(2011,1,1), max_value = datetime.today() - timedelta(days = 1))
    end_date = backtest_timeframe.date_input('Ending Date', value = datetime.today() - timedelta(days = 1), min_value = dt(2011,1,1), max_value = datetime.today() - timedelta(days = 1))
    
symbol = st.sidebar.selectbox('Crypto', scripts)
ticker = map_crypto2code[symbol]
    
indicator = st.sidebar.selectbox('Technical indicator', list(map_tech2fun.keys()))


# data = get_historical_prices(ticker)
# data = data[(data.index>=start_date)&(data.index<=end_date)]
data = pd.read_csv('artifacts/data_sample.csv')
data['Date'] = data.Date.astype('datetime64[ns]').dt.date
data = data.set_index('Date')
data = data[(data.index>=start_date)&(data.index<=end_date)]
   

ta_function = eval(map_tech2fun[indicator])
entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = ta_function(st, data, start_date, end_date)

  
st.sidebar.markdown('')
cf_bt = st.sidebar.button('Run simulation')
if cf_bt == False:
    st.info('Press run to simulate trading and visualise results.')    
if free_plan:
    if (cf_bt == True) and (ticker not in free_plan_list['cryptos']):
        st.info(ticker + ' crypto only available with the PRO plan.')
    if (cf_bt == True) and (indicator not in free_plan_list['technicals']):
        st.info(indicator + ' indicator only available with the PRO plan.')    
if (cf_bt == True) and \
    ((free_plan==False) \
         or \
    ((ticker in free_plan_list['cryptos']) and (indicator in free_plan_list['technicals']))):
    backtestdata = data.copy()
    if entry_comparator == 'LOWER THAN' and exit_comparator == 'LOWER THAN':
        buy_price, sell_price, strategy_signals = crossingdown_crossingdown(backtestdata, entry_data1, entry_data2, exit_data1, exit_data2)
    elif entry_comparator == 'LOWER THAN' and exit_comparator == 'HIGHER THAN':
        buy_price, sell_price, strategy_signals = crossingdown_crossingup(backtestdata, entry_data1, entry_data2, exit_data1, exit_data2) 
    elif entry_comparator == 'LOWER THAN' and exit_comparator == 'EQUAL TO':
        buy_price, sell_price, strategy_signals = crossingdown_equalto(backtestdata, entry_data1, entry_data2, exit_data1, exit_data2)
    elif entry_comparator == 'HIGHER THAN' and exit_comparator == 'LOWER THAN':
        buy_price, sell_price, strategy_signals = crossingup_crossingdown(backtestdata, entry_data1, entry_data2, exit_data1, exit_data2)
    elif entry_comparator == 'HIGHER THAN' and exit_comparator == 'HIGHER THAN':
        buy_price, sell_price, strategy_signals = crossingup_crossingup(backtestdata, entry_data1, entry_data2, exit_data1, exit_data2)
    elif entry_comparator == 'HIGHER THAN' and exit_comparator == 'EQUAL TO':
        buy_price, sell_price, strategy_signals = crossingup_equalto(backtestdata, entry_data1, entry_data2, exit_data1, exit_data2)
    elif entry_comparator == 'EQUAL TO' and exit_comparator == 'HIGHER THAN':
        buy_price, sell_price, strategy_signals = equalto_crossingup(backtestdata, entry_data1, entry_data2, exit_data1, exit_data2)
    elif entry_comparator == 'EQUAL TO' and exit_comparator == 'LOWER THAN':
        buy_price, sell_price, strategy_signals = equalto_crossingdown(backtestdata, entry_data1, entry_data2, exit_data1, exit_data2)
    elif entry_comparator == 'EQUAL TO' and exit_comparator == 'EQUAL TO':
        buy_price, sell_price, strategy_signals = equalto_equalto(backtestdata, entry_data1, entry_data2, exit_data1, exit_data2)
    
    def get_plot(n):
        plt.plot(data.tail(n).Close.tolist(), label='close')
        plt.plot(entry_data1.tail(n).tolist(), label='ent1')
        plt.plot(entry_data2.tail(n).tolist(), label='ent2')
        plt.plot(exit_data1.tail(n).tolist(), label='ex1')
        plt.plot(exit_data2.tail(n).tolist(), label='ex2')
        plt.legend()
        plt.show()    
    
    position = []
    for i in range(len(strategy_signals)):
        if strategy_signals[i] > 1:
            position.append(0)
        else:
            position.append(0)
        
    for i in range(len(backtestdata.Close)):
        if strategy_signals[i] == 1:
            position[i] = 1
        elif strategy_signals[i] == -1:
            position[i] = 0
        else:
            position[i] = position[i-1]
    
    st.caption(f'BACKTEST  RESULTS  FROM  {start_date}  TO  {end_date}')

    st.markdown('')
    
    buy_hold = backtestdata.Close.pct_change().dropna()
    strategy = (position[1:] * buy_hold).dropna()
    strategy_returns_per = np.exp(np.log(strategy+1).sum()) - 1
    bh_returns_per = np.exp(np.log(buy_hold+1).sum()) - 1
    
    n_days = len(backtestdata)
    annualized_returns = 365 / n_days * strategy_returns_per
    
    buy_signals = pd.Series(buy_price).dropna()
    sell_signals = pd.Series(sell_price).dropna()
    total_signals = len(buy_signals) + len(sell_signals)
    
    max_drawdown = pas.max_dd(strategy)
    
    profit = []
    losses = []
    for i in range(len(strategy)):
        if strategy[i] > 0:
            profit.append(strategy[i])
        elif strategy[i] < 0:
            losses.append(strategy[i])
        else:
            pass
        
    profit_factor = pd.Series(profit).sum() / (abs(pd.Series(losses)).sum())
    
    strat_percentage, bh_percentage, annr = st.columns(3)
    strat_percentage = strat_percentage.metric(label = 'Strategy Profit Percentage', value = f'{round(strategy_returns_per*100,2)}%')
    bh_percentage = bh_percentage.metric(label = 'Buy/Hold Profit Percentage', value = f'{round(bh_returns_per*100,2)}%')
    annr = annr.metric(label = 'Annualized Return', value = f'{round(annualized_returns*100,2)}%')
    
    nos, md, pf = st.columns(3)
    nos = nos.metric(label = 'Total No. of Signals', value = f'{total_signals}')
    md = md.metric(label = 'Max Drawdown', value = f'{round(max_drawdown,2)}%')
    pf = pf.metric(label = 'Profit Factor', value = f'{round(profit_factor,2)}')
    
    key_visuals = st.expander('PERFORMANCE COMPARISON')
        
    # key_visuals.caption('Strategy Equity Curve')
    # key_visuals.area_chart(scr)
    
    scr = pd.DataFrame((1+strategy).cumprod()).rename(columns = {'Close':'Returns'})
    scr.index = strategy.index
    strategy_drawdown = ffn.core.to_drawdown_series(scr.Returns)
    bh_drawdown = ffn.core.to_drawdown_series((1+buy_hold).cumprod())
    strategy_drawdown.name, bh_drawdown.name = 'Strategy', 'Buy/Hold'
    frames = [strategy_drawdown, bh_drawdown]
    drawdown = pd.concat(frames, axis = 1)    
    
    key_visuals.markdown('')
    key_visuals.markdown('')
    
    key_visuals.caption('Buy/Hold Returns Comparison')
    bhr = pd.DataFrame((1+buy_hold).cumprod()).rename(columns = {'Close':'Buy/Hold'})
    bhr.index = strategy.index
    scr = scr.rename(columns = {'Returns':'Strategy'})
    frames = [bhr, scr]
    bhr_compdf = pd.concat(frames, axis = 1)
    # key_visuals.line_chart(bhr_compdf)   
    
    
    bhr_compdf.columns=[symbol,'Your Strategy']
    bhr_compdf = pd.melt(bhr_compdf, ignore_index=False)
    bhr_compdf = bhr_compdf.reset_index()
    bhr_compdf.columns = ['Date', '$1,000 Portfolio Performance', 'Portfolio Value']
    bhr_compdf['Portfolio Value'] = bhr_compdf['Portfolio Value']*1000
    bhr_compdf['Portfolio Value'] = bhr_compdf['Portfolio Value'].astype(int)
   
    performance_plot = alt.Chart(bhr_compdf) \
    .mark_line() \
    .encode(
        x=alt.X("Date", axis=alt.Axis(format="%Y/%m")),
        y='Portfolio Value',
        color=alt.Color('$1,000 Portfolio Performance', legend=alt.Legend(orient="top")),
    )     
    
    key_visuals.altair_chart(performance_plot, use_container_width=True)  
    
    key_visuals.markdown('')
    key_visuals.markdown('')
    
    # key_visuals.caption('Maximum Drawdown')    
    # key_visuals.line_chart(drawdown)
    
    drawdown_details = st.expander('DRAWDOWN DETAILS')
    
    dd_details = ffn.core.drawdown_details(strategy)
    dd_details = proc_dd_details(dd_details, data)
    drawdown_details.table(dd_details)
    
    ratios = st.expander('RATIOS')
    
    ratios.caption('Values Assumed:  Benchmark = Bitcoin,  Risk-Free Rate = 0.01')
    
    ratios.markdown('')
    
    sharpe = pas.sharpe_ratio(strategy, 0.01)
    calmar = pas.calmar_ratio(strategy, 0.01)
    sortino = sortino_ratio(strategy, 255, 0.01)
    
    sharpe_ratio, calmar_ratio, sortino_ratio = ratios.columns(3)
    sharpe_ratio = sharpe_ratio.metric(label = 'Sharpe Ratio', value = f'{round(sharpe,3)}')
    calmar_ratio = calmar_ratio.metric(label = 'Calmar Ratio', value = f'{round(calmar,3)}')
    sortino_ratio = sortino_ratio.metric(label = 'Sortino Ratio', value = f'{round(sortino,3)}')
    
    benchmark_data = get_historical_prices('BTC')
    benchmark_data = benchmark_data[(benchmark_data.index>=start_date)&(benchmark_data.index<=end_date)]
    benchmark = benchmark_data.Close.pct_change().dropna()
    
    treynor = pas.treynor_ratio(strategy, benchmark, 0.01)
    information = pas.information_ratio(strategy, benchmark)
    modigliani = pas.modigliani_ratio(strategy, benchmark, 0.01)
    
    treynor_ratio, information_ratio, modigliani_ratio = ratios.columns(3)
    treynor_ratio = treynor_ratio.metric(label = 'Treynor Ratio', value = f'{round(treynor,3)}')
    information_ratio = information_ratio.metric(label = 'Information Ratio', value = f'{round(information,3)}')
    modigliani_ratio = modigliani_ratio.metric(label = 'Modigliani Ratio', value = f'{round(modigliani,3)}')
    
    sterling = pas.sterling_ratio(strategy, 0.01, 5) 
    burke = pas.burke_ratio(strategy, 0.01, 5)
    cond_sharpe = pas.conditional_sharpe_ratio(strategy, 0.01, 0.05)

    sterling_ratio, burke_ratio, cond_sharpe_ratio = ratios.columns(3)
    sterling_ratio = sterling_ratio.metric(label = 'Sterling Ratio', value = f'{round(sterling,3)}')
    burke_ratio = burke_ratio.metric(label = 'Burke Ratio', value = f'{round(burke,3)}')
    cond_sharpe_ratio = cond_sharpe_ratio.metric(label = 'Conditional Sharpe Ratio', value = f'{round(cond_sharpe,3)}')
    
    general_statistics = st.expander('GENERAL STATISTICS')
    
    strategy_df = pd.DataFrame(strategy).rename(columns = {'Close':'Strategy'})
    buy_hold_df = pd.DataFrame(buy_hold).rename(columns = {'Close':'Buy/Hold'})
    benchmark_df = pd.DataFrame(benchmark).rename(columns = {'Close':'Benchmark'})
    
    frames = [strategy_df, buy_hold_df, benchmark_df]
    stats_df = pd.concat(frames, axis = 1)
    
    general_stats = pat.stats_table(stats_df, manager_col = 0, other_cols = [1,2])
    general_statistics.table(general_stats)
    
    advanced_statistics = st.expander('ADVANCED STATISTICS')
    
    advanced_stats = pat.create_downside_table(stats_df, [0,1,2])
    advanced_statistics.table(advanced_stats)
