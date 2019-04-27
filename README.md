# Stock Price Prediction

This is our group project of the course「Machine Learning」. The main idea is to use neural network to do price prediction as well as to learn the features of the stock market.
<br />
## Change-Log
<br />
### 2019.04.26
1. At the momoent the implemented algorithms are **rnn**, **lstm**, **lstm-peephole** and **gru**. To compare RNNs with some other algorithms, I include another two algorithms, averaging and EMA.

<br />

## TO DO
1. At first glance, it seems averaging and EMA work fine with price prediction.But actually they are not. For stock price prediction, the point is not to make only the next day prediction but several-day prediction. We can make EMA prediction in windows in order to expose its drawbacks.
2. The current training speed is very slow. User Momentum and AdaGrad to speed up.
3. For the current dataset, we don't have many features. But we could give 'Linear Regression' a try.





Refs:

1. <a href="https://www.datacamp.com/community/tutorials/lstm-python-stock-market#download">Stock Market Predictions with LSTM in Python</a><br />
2. <a href="https://www.kaggle.com/raoulma/ny-stock-price-prediction-rnn-lstm-gru">NY Stock Price Prediction RNN LSTM GRU</a>

