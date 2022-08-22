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
import matplotlib.pyplot as plt 
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px

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
    port_strat = np.cumproduct(position[1:] * buy_hold+1)
    port_symbol = np.cumproduct(buy_hold+1)
    strategy_returns_per = np.exp(np.log(strategy+1).sum()) - 1
    bh_returns_per = np.exp(np.log(buy_hold+1).sum()) - 1
    
    n_days = len(backtestdata)
    annualized_returns = 365 / n_days * strategy_returns_per
    
    buy_signals = pd.Series(buy_price).dropna()
    sell_signals = pd.Series(sell_price).dropna()
    total_signals = len(buy_signals) + len(sell_signals)
    
    max_drawdown = ta.max_drawdown(port_strat)
    sharpe = ta.sharpe_ratio(port_strat)    
    sortino = ta.sortino_ratio(port_strat)
    jensen = ta.jensens_alpha(ta.performance.percent_return(port_strat).dropna(), 
                              ta.performance.percent_return(port_symbol).dropna())
    # jensens_alpha_metric = ta.jensens_alpha(port_strat)    
    # variance = ta.statistics.variance(port_strat,port_strat.shape[0]).tail(1).values[0]    
    # kurtosis = ta.statistics.kurtosis(port_strat,port_strat.shape[0]).tail(1).values[0]    

    
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
    
    strat_percentage, bh_percentage, md, sharpe_ratio, sortino_metric, jensen_metric = st.columns(6)
    bh_percentage = bh_percentage.metric(label = symbol + ' Performance', value = f'{round(bh_returns_per*100,2)}%')    
    strat_percentage = strat_percentage.metric(label = 'Strategy Performance', value = f'{round(strategy_returns_per*100,2)}%')
    sharpe_ratio = sharpe_ratio.metric(label = 'Sharpe Ratio', value = f'{round(sharpe,3)}')
    md = md.metric(label = 'Drawdown', value = f'{round(max_drawdown,2)}%')
    sortino_metric = sortino_metric.metric(label = 'Sortino Ratio', value = f'{round(sortino,2)}')
    jensen_metric = jensen_metric.metric(label = "Jensen's Alpha", value = f'{round(jensen,3)}')
    
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

    # df = px.data.stocks()
    df = pd.concat(frames, axis = 1).reset_index()
    df.columns = ['date',symbol,'Strategy']
    df[[symbol,'Strategy']]=df[[symbol,'Strategy']]*1000
    fig = px.line(df, x="date", y=df.columns,
                  hover_data={"date": "|%B %d, %Y"},
                  title='$1,000 Portfolio Performance')
    fig.update_xaxes(
        dtick="M1",
        tickformat="%b\n%Y")    
    fig.update_xaxes(rangeslider_visible=True) 
    # fig.write_html('abc.html')
    st.plotly_chart(fig, use_container_width=True)  
    
    
        
    
    key_visuals.markdown('')
    key_visuals.markdown('')
    
    # key_visuals.caption('Maximum Drawdown')    
    # key_visuals.line_chart(drawdown)
    
    drawdown_details = st.expander('DRAWDOWN DETAILS')
    
    dd_details = ffn.core.drawdown_details(strategy)
    dd_details = proc_dd_details(dd_details, data)
    drawdown_details.table(dd_details)
    
    def NormalizeData(data):
        return (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))    
    data_buysell = backtestdata.copy()[['Close']].reset_index()
    
    data_buysell['BuyPrice'] = buy_price
    sell_price_ = sell_price.copy()
    try:
        if strategy_signals.index(-1)<strategy_signals.index(1):
            sell_price_[strategy_signals.index(-1)] = np.nan
    except:
        pass
    data_buysell['SellPrice'] = sell_price_
    data_buysell['Signal1'] = entry_data1.values
    data_buysell['Signal2'] = entry_data2.values
    data_buysell['Signal3'] = exit_data1.values
    data_buysell['Signal4'] = exit_data2.values


    df = backtestdata.merge(data_buysell)
    fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                    open=df['Open'], high=df['High'],
                    low=df['Low'], close=df['Close'],
                    name=symbol)
                          ])
    for s in ['Signal1','Signal2','Signal3','Signal4']:
        fig.add_trace(go.Scatter(x=df['Date'], y=df[s],
                            mode='lines',
                            name=s))    
    for buy_days in df[['Date','BuyPrice']].dropna().Date.tolist():
        fig.add_vline(name='Buy signal', x=buy_days, line_width=3, line_dash="dash", line_color="green")
    for sell_days in df[['Date','SellPrice']].dropna().Date.tolist():
        fig.add_vline(name='Sell signal', x=sell_days, line_width=3, line_dash="dash", line_color="red")

    fig.update_layout(
        title='Strategy Signals',
        yaxis_title=symbol
        )
    
    # fig.write_html('abc.html')  
    st.plotly_chart(fig, use_container_width=True) 
    
    # ratios = st.expander('RATIOS')
    
    # ratios.caption('Values Assumed:  Benchmark = Bitcoin,  Risk-Free Rate = 0.01')
    
    # ratios.markdown('')
    
    # sharpe = pas.sharpe_ratio(strategy, 0.01)    
    # calmar = pas.calmar_ratio(strategy, 0.01)
    # sortino = sortino_ratio(strategy, 255, 0.01)
    
    # sharpe_ratio, calmar_ratio, sortino_ratio = ratios.columns(3)
    # sharpe_ratio = sharpe_ratio.metric(label = 'Sharpe Ratio', value = f'{round(sharpe,3)}')
    # calmar_ratio = calmar_ratio.metric(label = 'Calmar Ratio', value = f'{round(calmar,3)}')
    # sortino_ratio = sortino_ratio.metric(label = 'Sortino Ratio', value = f'{round(sortino,3)}')
    
    # benchmark_data = get_historical_prices('BTC')
    # benchmark_data = benchmark_data[(benchmark_data.index>=start_date)&(benchmark_data.index<=end_date)]
    # benchmark = benchmark_data.Close.pct_change().dropna()
    
    # treynor = pas.treynor_ratio(strategy, benchmark, 0.01)
    # information = pas.information_ratio(strategy, benchmark)
    # modigliani = pas.modigliani_ratio(strategy, benchmark, 0.01)
    
    # treynor_ratio, information_ratio, modigliani_ratio = ratios.columns(3)
    # treynor_ratio = treynor_ratio.metric(label = 'Treynor Ratio', value = f'{round(treynor,3)}')
    # information_ratio = information_ratio.metric(label = 'Information Ratio', value = f'{round(information,3)}')
    # modigliani_ratio = modigliani_ratio.metric(label = 'Modigliani Ratio', value = f'{round(modigliani,3)}')
    
    # sterling = pas.sterling_ratio(strategy, 0.01, 5) 
    # burke = pas.burke_ratio(strategy, 0.01, 5)
    # cond_sharpe = pas.conditional_sharpe_ratio(strategy, 0.01, 0.05)

    # sterling_ratio, burke_ratio, cond_sharpe_ratio = ratios.columns(3)
    # sterling_ratio = sterling_ratio.metric(label = 'Sterling Ratio', value = f'{round(sterling,3)}')
    # burke_ratio = burke_ratio.metric(label = 'Burke Ratio', value = f'{round(burke,3)}')
    # cond_sharpe_ratio = cond_sharpe_ratio.metric(label = 'Conditional Sharpe Ratio', value = f'{round(cond_sharpe,3)}')
    
    # general_statistics = st.expander('GENERAL STATISTICS')
    
    # strategy_df = pd.DataFrame(strategy).rename(columns = {'Close':'Strategy'})
    # buy_hold_df = pd.DataFrame(buy_hold).rename(columns = {'Close':'Buy/Hold'})
    # benchmark_df = pd.DataFrame(benchmark).rename(columns = {'Close':'Benchmark'})
    
    # frames = [strategy_df, buy_hold_df, benchmark_df]
    # stats_df = pd.concat(frames, axis = 1)
    
    # general_stats = pat.stats_table(stats_df, manager_col = 0, other_cols = [1,2])
    # general_statistics.table(general_stats)
    
    # advanced_statistics = st.expander('ADVANCED STATISTICS')
    
    # advanced_stats = pat.create_downside_table(stats_df, [0,1,2])
    # advanced_statistics.table(advanced_stats)
