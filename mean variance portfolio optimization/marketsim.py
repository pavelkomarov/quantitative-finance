"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data
import matplotlib.pyplot as plt

def compute_portvals(orders_file, start_val=1000000, leverage_limit=2.0):
    orders = pd.read_csv(orders_file)
    orders = orders.sort_values('Date')
    orders.index = range(len(orders))

    #I can assume orders are in order because I sorted them. Now get dates.
    start_date = dt.datetime.strptime(orders.ix[0,'Date'], '%Y-%m-%d')
    end_date = dt.datetime.strptime(orders.ix[len(orders)-1,'Date'], '%Y-%m-%d')
    dates = pd.date_range(start_date, end_date)

    #pull out all the data I could possibly need
    syms = list(set(orders.ix[:,'Symbol']))
    prices_all, syms = get_data(syms, dates)# automatically adds SPY

    portvals = []
    cash = start_val
    holdings = {sym:0 for sym in syms}
    for date in prices_all.index:
        #logic to update holdings and cash
        datestr = date.strftime('%Y-%m-%d')
        todaysbusiness = orders[orders['Date']==datestr]
        for i in todaysbusiness.index:#If we made any orders on this date
            preleverage = findLeverage(cash, holdings, prices_all, date)

            #find what trade we're supposed to be making and update stuff
            sym = todaysbusiness.ix[i,'Symbol']
            shares = todaysbusiness.ix[i,'Shares']
            if todaysbusiness.ix[i,'Order']=='SELL': shares = -shares
            holdings[sym] = holdings[sym] + shares
            cash = cash - shares*prices_all.ix[date,sym]
            
            postleverage = findLeverage(cash, holdings, prices_all, date)
            #if holding that stuff violates the leverage rules, don't do that stuff
            if (postleverage > leverage_limit and postleverage > preleverage):
                holdings[sym] = holdings[sym] - shares
                cash = cash + shares*prices_all.ix[date,sym]
            #else actually do the order

        #calculate total current value and add it to the history
        currentval = cash
        for sym in holdings:
            currentval = currentval + holdings[sym]*prices_all.ix[date,sym]
        portvals.append(currentval)

    #transform the list of values in to a dataframe and return
    return pd.DataFrame(portvals, index=prices_all.index, columns=['Value'])

def findLeverage(cash, holdings, prices_all, date):
    shortsval = 0
    longsval = 0
    for sym in holdings:
        if holdings[sym] < 0:
            shortsval = shortsval + holdings[sym]*prices_all.ix[date,sym]
        elif holdings[sym] > 0:
            longsval = longsval + holdings[sym]*prices_all.ix[date,sym]
    return (longsval - shortsval) / (longsval + shortsval + cash)

def simulate():
    # Process orders
    portvals = compute_portvals("in_sample.csv", start_val=10000, leverage_limit=1000)
    portvals = portvals[portvals.columns[0]]# just get the first column

    # Get portfolio stats
    # Get start and end date
    start_date = portvals.index[0]
    end_date = portvals.index[-1]

    # Get benchmark data over that period to compare
    dates = pd.date_range(start_date, end_date)
    prices_benchmark, syms = get_data(['$SPX'], dates)
    prices_benchmark = prices_benchmark[prices_benchmark.columns[1]]#transform to series instead of dataframe

    # calculate cumulative return for portfolio and SPX
    cum_ret = (portvals[-1] - portvals[0]) / portvals[0]
    cum_ret_benchmark = (prices_benchmark[-1] - prices_benchmark[0]) / prices_benchmark[0]

    # calculate average daily return
    portval_norm = portvals/portvals[0]
    benchmark_norm = prices_benchmark/prices_benchmark[0]
    portval_daily = portval_norm.values[1:] / portval_norm.values[:-1] - 1# (end - start)/start
    benchmark_daily = benchmark_norm.values[1:] / benchmark_norm.values[:-1] - 1# (end - start)/start
    avg_daily_ret = portval_daily.mean()
    avg_daily_ret_benchmark = benchmark_daily.mean()

    #calculate standard deviations
    std_daily_ret = portval_daily.std(ddof=1)
    std_daily_ret_benchmark = benchmark_daily.std(ddof=1)

    #calculate Sharpe ratio
    sharpe_ratio = avg_daily_ret/std_daily_ret*np.sqrt(252)
    sharpe_ratio_benchmark = avg_daily_ret_benchmark/std_daily_ret_benchmark*np.sqrt(252)

    # print out all the stats
    print "Date Range: {} to {}".format(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of benchmark : {}".format(sharpe_ratio_benchmark)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of benchmark : {}".format(cum_ret_benchmark)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of benchmark : {}".format(std_daily_ret_benchmark)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of benchmark : {}".format(avg_daily_ret_benchmark)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])

    #plot the normalized portfolio performance against the benchmark
    plotframe = pd.concat([portvals/portvals[0], prices_benchmark/prices_benchmark[0]], \
        keys=['Portfolio', '$SPX'], axis=1)
    plotframe.plot()#series
    plt.show()

if __name__ == "__main__":
    simulate()
