`quadprog.py` provides a helper method to make calling the quadratic program solver from cvxopt easier.

`util.py` has a helper function for reading data. As I add more things to this repo, this may end up somewhere more general.

Run `python3 MVPO.py` to see an example with only a few tickers. With the whole S&P500 it takes a few minutes to run.

I originally took inspiration from an article, the url of which now forwards [here](https://www.quantopian.com/posts/the-efficient-frontier-markowitz-portfolio-optimization-in-python).
