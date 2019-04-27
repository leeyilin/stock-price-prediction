import numpy as np
import pandas as pd
import math
import sklearn
import sklearn.preprocessing
import datetime
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import keras


def my_pltsavefig(fig_title, mydir='./figure/', fig_extension='.png'):
    filename = mydir + fig_title.replace(' ', '-') + fig_extension
    if not os.path.exists(filename):
        plt.savefig(filename)
        return True
    else:
        return False


def analyze_data(df):
    print(df.info())
    print(df.head())
    print(df.tail())
    print(df.describe())
    simple_plot(df)


def simple_plot(df, plot_volume=True):
    plt.figure(figsize=(15, 5))
    plt.plot(df.open.values, color='red', label='open')
    plt.plot(df.close.values, color='green', label='close')
    plt.plot(df.low.values, color='blue', label='low')
    plt.plot(df.high.values, color='black', label='high')
    title = 'stock price overview'
    plt.title(title)
    plt.xlabel('time [days]')
    plt.ylabel('price')
    plt.legend(loc='best')
    my_pltsavefig(title)
    plt.show()


class ModelData(object):
    MODEL_RNN_BASIC = 'basic-rnn'
    MODEL_RNN_LSTM = 'lstm'
    MODEL_RNN_LSTM_PEEPHOLE = 'lstm-peephole'
    MODEL_RNN_GRU = 'gru'
    ALL_MODELS = [MODEL_RNN_BASIC, MODEL_RNN_LSTM, 
            MODEL_RNN_LSTM_PEEPHOLE, MODEL_RNN_GRU]

    def __init__(self, df, model_name, 
            n_layers=2, n_neurons=200, seq_len=20):
        self.df = df
        self.seq_len = seq_len
        self.normalize_data()
        self.x_train, self.y_train, self.x_valid, \
                self.y_valid, self.x_test, self.y_test = self.dataset_division()
        self.index_in_epoch = 0
        self.perm_array = np.arange(self.x_train.shape[0])
        np.random.shuffle(self.perm_array)
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        if model_name == ModelData.MODEL_RNN_BASIC:
            self.layers = [tf.keras.layers.SimpleRNNCell(
                units = self.n_neurons, activation=tf.nn.elu)
                for _ in range(self.n_layers)]
        elif model_name == ModelData.MODEL_RNN_LSTM:
            self.layers = [tf.keras.layers.LSTMCell(
                units = self.n_neurons, activation=tf.nn.elu)
                for _ in range(self.n_layers)]
        elif model_name == ModelData.MODEL_RNN_LSTM_PEEPHOLE:
            self.layers = [tf.keras.layers.LSTMCell(
                units = self.n_neurons, activation=tf.nn.leaky_relu,
                use_peepholes = True)
                for _ in range(self.n_layers)]
        elif model_name == ModelData.MODEL_RNN_GRU:
            self.layers = [tf.keras.layers.GRUCell(
                units = self.n_neurons, activation=tf.nn.leaky_relu)
                for _ in range(self.n_layers)]
        else:
            raise ValueError('illegal model-name!')

        self.y_train_pred = None
        self.y_valid_pred = None
        self.y_test_pred = None

    def normalize_data(self):
        min_max_scaler = sklearn.preprocessing.MinMaxScaler()
        cols = ['open', 'high', 'low', 'close']
        for col in cols:
            self.df[col] = min_max_scaler.fit_transform(
                    self.df[col].values.reshape(-1, 1))


    def dataset_division(self, valid_set_size_percentage = 10,
            test_set_size_percentage = 10):
        # transform to numpy array.
        data_raw = self.df.values
        data = []
        for index in range(len(data_raw) - self.seq_len):
            # a 3D array.
            data.append(data_raw[index: index + self.seq_len])
        data = np.array(data)
        valid_set_size = int(np.round(
            valid_set_size_percentage / 100 * data.shape[0]))
        test_set_size = int(np.round(
            test_set_size_percentage / 100 * data.shape[0]))
        train_set_size = data.shape[0] - (valid_set_size + test_set_size)
        x_train = data[:train_set_size, :-1, :]
        y_train = data[:train_set_size, -1, :]
        x_valid = data[train_set_size : \
                train_set_size + valid_set_size, :-1, :]
        y_valid = data[train_set_size : \
                train_set_size + valid_set_size, -1, :]
        x_test = data[train_set_size + valid_set_size : , :-1, :]
        y_test = data[train_set_size + valid_set_size : , -1, :]
        return x_train, y_train, x_valid, y_valid, x_test, y_test

    def train(self):

        def get_next_batch(batch_size):
            start = self.index_in_epoch
            self.index_in_epoch += batch_size
            if self.index_in_epoch > self.x_train.shape[0]:
                np.random.shuffle(self.perm_array)
                start = 0
                self.index_in_epoch = batch_size
            end = self.index_in_epoch
            return self.x_train[self.perm_array[start: end]], \
                    self.y_train[self.perm_array[start: end]]

        n_steps = self.seq_len - 1
        n_inputs = 4
        n_outputs = 4
        learning_rate = 0.001
        batch_size = 50
        n_epochs = 100
        train_set_size = self.x_train.shape[0]
        test_set_size = self.x_test.shape[0]
        print('train set size: %d, test set size: %d' % \
                (train_set_size, test_set_size))

        tf.reset_default_graph()

        X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
        y = tf.placeholder(tf.float32, [None, n_outputs])

        multi_layer_cell = tf.keras.layers.StackedRNNCells(self.layers)
        #multi_layer_cell = tf.contrib.rnn.MultiRNNCell(self.layers)
        rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, 
                X, dtype=tf.float32)
        stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, self.n_neurons])
        stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)
        outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])
        outputs = outputs[:, n_steps - 1, :]

        loss = tf.reduce_mean(tf.square(outputs - y))
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        training_op = optimizer.minimize(loss)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for iteration in range(n_epochs * train_set_size // batch_size):
                x_batch, y_batch = get_next_batch(batch_size)
                sess.run(training_op, feed_dict = {X: x_batch, y: y_batch})
                if iteration % (5 * train_set_size // batch_size) == 0:
                    mse_train = loss.eval(
                            feed_dict = {X: self.x_train, y: self.y_train})
                    mse_valid = loss.eval(
                            feed_dict = {X: self.x_valid, y: self.y_valid})
                    print('%d epochs: MSE train/valid = %.6f/%.6f' %\
                            (math.ceil(iteration * batch_size / train_set_size), 
                                mse_train, mse_valid))

            self.y_train_pred = sess.run(outputs, feed_dict = {X: self.x_train})
            self.y_valid_pred = sess.run(outputs, feed_dict = {X: self.x_valid})
            self.y_test_pred = sess.run(outputs, feed_dict = {X: self.x_test})

    def visualize(self):
        # 0 = open, 1 = close, 2 = high, 3 = low
        ft = 0
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)

        plt.plot(np.arange(self.y_train.shape[0]), self.y_train[:, ft], 
                color = 'blue', label = 'train target')
        plt.plot(np.arange(self.y_train.shape[0], 
            self.y_train.shape[0] + self.y_valid.shape[0]), self.y_valid[:, ft], 
            color = 'gray', label = 'valid target')
        plt.plot(np.arange(self.y_train.shape[0] + self.y_valid.shape[0],
            self.y_train.shape[0] + self.y_valid.shape[0] + self.y_test.shape[0]), 
            self.y_test[:, ft], color = 'black', label = 'test target')
        plt.plot(np.arange(self.y_train_pred.shape[0]), self.y_train_pred[:, ft], 
                color = 'red', label = 'train prediction')
        plt.plot(np.arange(self.y_train_pred.shape[0], 
            self.y_train_pred.shape[0] + self.y_valid_pred.shape[0]), 
            self.y_valid_pred[:, ft], color = 'orange', label = 'valid prediction')
        plt.plot(np.arange(self.y_train_pred.shape[0] + self.y_valid_pred.shape[0],
            self.y_train_pred.shape[0] + self.y_valid_pred.shape[0]
            + self.y_test_pred.shape[0]),
            self.y_test_pred[:, ft], color = 'green', label = 'test prediction')
        plt.title('past and future stock prices')
        plt.xlabel('time [days]')
        plt.ylabel('normalized price')
        plt.legend(loc = 'best')

        plt.subplot(1, 2, 2)
        plt.plot(np.arange(self.y_train.shape[0], 
            self.y_train.shape[0] + self.y_test.shape[0]),
                self.y_test[:, ft], color = 'black', label = 'test target')
        plt.plot(np.arange(self.y_train_pred.shape[0], 
            self.y_train_pred.shape[0] + self.y_test_pred.shape[0]), 
            self.y_test_pred[:, ft], color = 'green', label = 'test prediction')
        plt.title('future stock prices')
        plt.xlabel('time [days]')
        plt.ylabel('normalized price')
        plt.legend(loc = 'best')

        corr_price_development_train = np.sum(np.equal(
            np.sign(self.y_train[:, 1] - self.y_train[:, 0]), 
            np.sign(self.y_train_pred[:, 1] - self.y_train_pred[:, 0]))\
                    .astype(int)) / self.y_train.shape[0]
        corr_price_development_valid = np.sum(np.equal(
            np.sign(self.y_valid[:, 1] - self.y_valid[:, 0]), 
            np.sign(self.y_valid_pred[:, 1] - self.y_valid_pred[:, 0])).\
                    astype(int)) / self.y_valid.shape[0]
        corr_price_development_test = np.sum(np.equal(
            np.sign(self.y_test[:, 1] - self.y_test[:, 0]), 
            np.sign(self.y_test_pred[:, 1] - self.y_test_pred[:, 0])).\
                    astype(int)) / self.y_test.shape[0]
        
        print('correct sign prdiction for close - ' + \
                'open price for train/valid/test: %.2f/%.2f/%.2f' % (
                    corr_price_development_train, corr_price_development_valid,
                    corr_price_development_test))
        my_pltsavefig(fig_title='stock price prediction by RNN')
        plt.show()

    def predict_by_averaging(self):
        df_close = self.df.close
        window_size = 20
        N = df_close.shape[0]
        std_avg_predictions = []
        mse_errors = []
        for i in range(window_size, N):
            std_avg_predictions.append(np.mean(df_close[i - window_size : i]))
            mse_errors.append((std_avg_predictions[-1] - df_close[i])**2)
        print('MSE error for standard averaging: %.5f' % (0.5 * np.mean(mse_errors)))

        plt.figure(figsize=(15, 5))
        plt.plot(np.arange(window_size, N), df_close.values[window_size:], 
                color='green', label='target')
        plt.plot(np.arange(window_size, N), np.array(std_avg_predictions), 
                color='red', label='prediction')
        title = 'stock price prediction by averaging'
        plt.title(title)
        plt.ylabel('close price')
        plt.legend(loc='best')
        my_pltsavefig(title)
        plt.show()

    def predict_by_ema(self):
        df_close = self.df.close
        N = df_close.shape[0]
        running_mean = 0.0
        avg_predictions = [running_mean]
        mse_errors = []
        decay = 0.5
        for i in range(1, N):
            running_mean = running_mean * decay + (1 - decay) * df_close[i - 1]
            avg_predictions.append(running_mean)
            mse_errors.append((avg_predictions[-1] - df_close[i]) ** 2)
        print('MSE error for EMA averaging: %.5f' % (0.5 * np.mean(mse_errors)))

        plt.figure(figsize=(15, 5))
        plt.plot(np.arange(N), df_close.values, 
                color='green', label='target')
        plt.plot(np.arange(N), np.array(avg_predictions), 
                color='red', label='prediction')
        title = 'stock price prediction by ema'
        plt.title(title)
        plt.ylabel('close price')
        plt.legend(loc='best')
        my_pltsavefig(title)
        plt.show()


def load_data(filename):
    df = pd.read_csv(filename, index_col = 1)
    drop_cols = ['index_code', 'volume', 'money', 'change', 'label']
    for col in drop_cols:
        df.drop([col], 1, inplace=True)
    cols = list(df.columns.values)
    print('df.columns.values = ', cols)
    analyze_data(df)
    return df


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    df = load_data(filename='./data/SH000001.csv')
    for model_name in ModelData.ALL_MODELS:
        md = ModelData(df.copy(), model_name)
        md.train()
        md.visualize()
        break

    md.predict_by_averaging()
    md.predict_by_ema()

