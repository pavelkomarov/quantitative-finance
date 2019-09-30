import pandas as pd

## Read in data from the data folder
# symbols, A list of tickers
# dates, A pandas date_range to fetch data over
# allinfo, Whether to keep all columns or only return date and close price
def get_data(symbols, dates, allinfo=False):
	df = pd.DataFrame(index=dates)
	spy = 'SPY' in symbols
	if spy: symbols.remove('SPY')#it's going to get added for comparison anyway, so no double-adding

	if allinfo:
		df_temp = pd.read_csv("../data/SPY.csv", index_col='Date', parse_dates=True, na_values=['nan'])
		df_temp.columns = ["SPY "+header for header in df_temp.columns]
		df = df.join(df_temp)#left join to ensure order is preserved
		df = df.dropna(subset=["SPY Adj Close"])#but what we really wanted was an inner join

		for symbol in symbols:
			df_temp = pd.read_csv("../data/"+str(symbol)+".csv", \
				index_col='Date', parse_dates=True, na_values=['nan'])
			df_temp.columns = [symbol+" "+header for header in df_temp.columns]
			df = df.join(df_temp)
	else:
		df_temp = pd.read_csv("../data/SPY.csv", index_col='Date',
					parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
		df_temp = df_temp.rename(columns={'Adj Close': 'SPY'})
		df = df.join(df_temp)
		df = df.dropna(subset=["SPY"])
		
		for symbol in symbols:
			df_temp = pd.read_csv("../data/"+str(symbol)+".csv", index_col='Date',
					parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
			df_temp = df_temp.rename(columns={'Adj Close': symbol})
			df = df.join(df_temp)
				
	df = df.fillna(method='ffill', axis=0)# fill in missing data (days not traded)
	df = df.fillna(method='bfill', axis=0)# backfill with stock's first trade
	df = df.dropna(axis=1, how='all')#drop any column that is still full of NaNs

	if allinfo: remsyms = [x[:-5] for x in list(df.columns)[1::6]]#symbols not wiped out by dropna
	else: remsyms = [x for x in list(df.columns)[:]]
	if not spy: remsyms.remove('SPY')#include SPY or don't based on whether it was passed in

	return df, remsyms
