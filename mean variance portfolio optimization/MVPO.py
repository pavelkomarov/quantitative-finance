import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from util import get_data
from quadprog import quadprog

#Mean Variance Portfolio Optimizer
class MVPO(object):

	def __init__(self, syms):
		self.syms = syms # the things we will consider adding to the portfolio

	## find portfolio with best Sharpe ratio over some date range (predictive of future? dubious.)
	# http://blog.quantopian.com/markowitz-portfolio-optimization-2/
	# @param dates, A pandas date_range
	# @return wstar, The optimal "weighting" (distribution) of the portfolio across the equities in syms
	def optimize(self, dates):
		data, self.syms = get_data(self.syms, dates)
		avg_daily_rets, C, std_devs = self.get_daily_stats(data)
		frontier_risks, frontier_returns, opt, wstar = self.binary_search(0, 1000, avg_daily_rets, C)

		print("leverage of optimal portfolio =", np.abs(wstar).sum())
		self.plot_all(std_devs, avg_daily_rets, frontier_risks, frontier_returns, opt)
		
		return wstar

	#do the optimization in a binary-search configuration, looking for best Sharpe ratio
	def binary_search(self, l, r, avg_daily_rets, C):
		N = len(self.syms)
		opt_params = self.get_opt_arrays(avg_daily_rets, C)

		frontier_returns = []
		frontier_risks = []
		sharperatios = [0,0]

		mid = (l+r)/2#a peak-finding algorithm
		while l < r:
			for i in [0, 1]:#find slope of frontier at mid and at mid+1
				opt_params[5][0,0] = -(mid+i)*0.00001#how the QP encodes "find least-volatile at ___ return"
				print(-opt_params[5][0,0])#so I can see what the search is doing on the command line
				w, success = quadprog(*opt_params)
				if success:
					x = np.sqrt(w[0:N].T.dot(C.dot(w[0:N])))[0,0]#ugly because numpy is stupid. It's just w'Cw and rets'w
					y = avg_daily_rets.T.dot(w[0:N])[0,0]#w[0:N] contains the wieghts. The others are auxiliary to help with abs() in QP.
					frontier_risks.append(x)
					frontier_returns.append(y)
					sharperatios[i] = y/x
				else:
					sharperatios[i] = -i#so if I run off the upper end I get [0,-1], which tells it to bring right inward

			if sharperatios[0] > sharperatios[1]:#getting less steep (flattening)
				r = mid
			elif sharperatios[0] < sharperatios[1]:#getting more steep (still rising)
				l = mid+1
			mid = (r + l)/2
		
		#find best according to what mid is returned
		opt_params[5][0,0] = -mid*0.00001
		wstar, success = quadprog(*opt_params)#should be successful
		
		frontier_risks.sort()
		frontier_returns.sort()
		opt = (np.sqrt(wstar[0:N].T.dot(C.dot(wstar[0:N])))[0,0], avg_daily_rets.T.dot(wstar[0:N])[0,0])

		return frontier_risks, frontier_returns, opt, wstar[0:N]

	#set up to do min w'Cw - avg_ret'w
	#		s.t. -avg_ret'w <= -min_return (return better than this guaranteed))
	#			sum(abs(w)) <= 2 (leverage limit)
	#			sum(w) >= 1
	#All subtle equalities and inequalities are defined here, including the complicated auxiliary
	#parameters to make the absolute value constraint in to ordinary constraints.
	#https://www.wpi.edu/Pubs/E-project/Available/E-project-042707-112035/unrestricted/TurnoverConstraintsMQP.pdf
	#https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf
	def get_opt_arrays(self, avg_daily_rets, C):
		N = len(self.syms)
		H = np.r_[np.c_[C,np.zeros((N,2*N))], np.zeros((2*N,3*N))]#H = [C 0; 0 0]
		f = np.r_[-avg_daily_rets, np.zeros((2*N,1))]
		A = np.r_[np.c_[np.zeros((N,N)), -np.eye(N), np.zeros((N,N))], \
			np.c_[np.zeros((N,2*N)), -np.eye(N)], \
			np.c_[np.zeros((1,N)), np.ones((1,2*N))], \
			np.c_[-np.ones((1,N)), np.zeros((1,2*N))]]#A = [-avg_daily_rets' 0 0; 0 -I 0; 0 0 -I; 0 1' 1'; -1' 0 0]
		b = np.r_[np.zeros((2*N,1)), np.c_[2.0], np.c_[-1.0]]#b = [mu; 0; levlim+; -levlim-]
		Aeq = np.r_[np.c_[-avg_daily_rets.T, np.zeros((1,2*N))], \
			np.c_[-np.eye(N), np.eye(N), -np.eye(N)]]#Aeq = [;-I I -I]
		beq = np.zeros((N+1,1))
		return (H, f, A, b, Aeq, beq)

	#put all daily returns in a numpy array, and use that to get covariance and such
	def get_daily_stats(self, data):
		N = len(self.syms)
		daily_rets = np.zeros((N, len(data)-1))
		for i in range(N):
			adjcl = data[self.syms[i]]
			daily_rets[i] = adjcl.values[1:]/adjcl.values[:-1] - 1

		avg_daily_rets = np.mean(daily_rets, axis=1).reshape((N,1))#in same order as self.syms
		C = np.cov(daily_rets)
		std_devs = np.diag(np.sqrt(C)).reshape((N,1))

		return avg_daily_rets, C, std_devs

	#vizualization function
	def plot_all(self, std_devs, avg_daily_rets, frontier_risks, frontier_returns, opt):
		longs = []
		shorts = []
		for i in range(len(self.syms)):
			if avg_daily_rets[i,0]>0:#longs
				longs.append((std_devs[i,0], avg_daily_rets[i,0], self.syms[i]))
			else:#shorts
				shorts.append((std_devs[i,0], -avg_daily_rets[i,0], self.syms[i]))

		plt.scatter([pos[0] for pos in longs], [pos[1] for pos in longs], c='b', label='long')
		for i in range(len(longs)): plt.annotate(longs[i][2], (longs[i][0], longs[i][1]))		
		plt.scatter([pos[0] for pos in shorts], [pos[1] for pos in shorts], c='r', label='short')
		for i in range(len(shorts)): plt.annotate(shorts[i][2], (shorts[i][0], shorts[i][1]))

		plt.plot(frontier_risks, frontier_returns, 'y-o', label='efficient frontier')
		plt.plot(opt[0], opt[1], 'g*', markersize=14, label='point with best Sharpe Ratio')

		plt.legend(loc='upper left');
		plt.title('efficient frontier in mean-std space')
		plt.xlabel('standard deviation of daily return')
		plt.ylabel('mean daily return')
		plt.show()

	#use the weights to make trades over the dates given. Simple rule of buy and hold.
	def make_trades(self, weights, dates, cash=10000, file='orders.csv'):
		data, self.syms = get_data(self.syms, dates)
		start_date = data.index[0]
		end_date = data.index[-1]

		orderslist = []
		for i in range(len(self.syms)):
			sym = self.syms[i]
			alloccash = weights[i,0]*cash
			shares = alloccash/data.ix[start_date,sym]
			if alloccash > 50:#only do an action if its magnitude is significant in some sense, since there are fees
				orderslist.append([start_date, sym, 'BUY', shares, 'long'])#buy long
				orderslist.append([end_date, sym, 'SELL', shares, 'exit'])#exit the long
			elif alloccash < -50:
				orderslist.append([start_date, sym, 'SELL', -shares, 'long'])#sell short
				orderslist.append([end_date, sym, 'BUY', -shares, 'exit'])#exit the short

		orders = pd.DataFrame(orderslist, columns=['Date','Symbol','Order','Shares','Position'])
		orders.to_csv(file, columns=['Date','Symbol','Order','Shares'], index=False)

if __name__ == "__main__":
	syms = ['IBM','MSFT','AAPL','GOOG','SPY','INTC','XOM', 'NVDA', 'NFLX']
	#sp500 = open("../data/Lists/sp5002012.txt")
	#syms = []
	#for company in sp500:
	#	syms.append(company.strip())

	optimizer = MVPO(syms)

	train_dates = pd.date_range(dt.datetime(2010,12,31), dt.datetime(2011,3,31))
	bestport = optimizer.optimize(train_dates)#now buy at the end of this period and hold for some period
	optimizer.make_trades(bestport, train_dates, 10000, 'in_sample.csv')

	test_dates = pd.date_range(dt.datetime(2011,3,31), dt.datetime(2011,6,30))
	optimizer.make_trades(bestport, test_dates, 10000, 'out_of_sample.csv')
