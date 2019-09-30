This repo contains a series of quantitative finance projects from a course I took in 2016 called Machine Learning for Trading plus some others I've worked on. Because the foundation for me was that class, projects are sort of sequential, numbered from least sophisticated to more sophisticated.



TODOs:
- Basic, clean technical analysis scripts complete with header comments about how to use them (full data pipeline)
- Trading strategies, some from ML for Trading, some common knowledge, some to be invented, implemented where possible as runnable Jupyter notebooks with explanations and instructions.
  - A basic classification strategy to decide purely whether something will go up or down on a daily basis, utilizing data of equities themselves, their siblings, and any other daily data streams we can think of.
  - A set of runnable, backtestable market-cycle timing strategies.
  - That order-book noise high frequency trader Vikram and I talked about making, to be plugged in to pro.coinbase.com.
  - That mean-variance ?? trading strategy I implemented on a whim a couple years ago, beyond the extra credit.
  - A reinforcement-learning based approach.
  - Research in to quantification of options risk/valuation strategies.
  - An options strategy that finds bargains by looking for odd-ones-out of sorted price order.
  - An LSTM-based prediction model, to get more intuition about recurrent nets.
- A set of supporting objects and utilities (a "framework") for to make the above goals coherent.

