# I was reading [this](http://www.philosophicaleconomics.com/2016/02/uetrend/) recently and was impressed how unemployment
# is a fairly reliable predictor of oncoming recession: When it starts to turn and crosses significantly over a moving
# average, it's time to be on alert. Otherwise history suggests there is usually no cause for concern, maybe because
# speculators gonna speculate with ever higher expectations until some big, often external shock to the system compels them
# to break the habit, and that shock can show up elsewhere in the economy before everyone is able to stop lying to themselves.
#
# Thoughout the article, the author references [charts in FRED](https://fred.stlouisfed.org/series/UNRATE). (Thanks St. Louis
# FED! <3) But FRED's beautiful, interactive, stackable, collectible-in-dashboards charts unfortunately provide no way to
# overlay trendlines. This is just a simple script to overlay a moving average line to bring the analysis performed in the
# article up to date. To make it work, you'll need the UNRATE data and [USREC](https://fred.stlouisfed.org/series/USREC) data.
#
# Todos:
# - Steal FRED's code (the <div id="zoom-and-share"> in view-source:https://fred.stlouisfed.org/series/USREC) so this is
#	interactive and can run in your browser.
# - Update FRED's code to support a trendline feature, preferrably as reuseable javascripts.
# - Since I expect a lot of this repo will end up being algorithmic trading sort of stuff in python, evolve this in to a
#	generic, configurable python timeseries plotting script.

import pandas
import numpy
import datetime
from matplotlib import pyplot

u = 'UNRATE' # which thing to use for plotting: absolute number or rate

## auxiliary to find moving average of array
# A array-like that can be sliced
# n number of entries to average over
def moving_average(A, n=12):
	t = numpy.cumsum(A, dtype=float)
	t[n:] = t[n:] - t[:-n]
	return t[n-1:] / n

# join unemployment data to recession data to get the relevant piece
f = pandas.read_csv(u + '.csv', index_col=0).join(pandas.read_csv('USREC.csv', index_col=0))

# data series
raw = f[u].values
rec = f['USREC'].values
mov12 = moving_average(raw)

# plot trends
pyplot.plot(raw, label='raw data')
pyplot.plot(range(len(raw)-len(mov12), len(raw)), mov12, label='12 month moving average')
# plot grey recession bars
for i, indicator in enumerate(rec):
	if indicator:
		pyplot.axvspan(i-1, i, facecolor='0.2', alpha=0.5, lw=0)
# label everything
pyplot.xlabel('time (monthly)')
if u == 'UNEMPLOYMENT':
	pyplot.ylabel('thousands of persons')
else:
	pyplot.ylabel('percent')
pyplot.xticks(numpy.arange(len(raw)//12)*12, [date[:-6] for date in list(f.index)[0::12]], rotation=60)
pyplot.title('Unemployment')
pyplot.legend()
pyplot.show()
