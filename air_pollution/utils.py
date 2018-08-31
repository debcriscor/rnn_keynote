
#
# Some functions to be used in the tutorial
#
# Developed by Debora Cristina Correa

import datetime
import pandas as pd
import matplotlib.pyplot as plt # for 2D plotting
import numpy as np


def parse_datetime(x):
	return datetime.datetime.strptime(x, '%Y %m %d %H')

def create_window(data, n_in = 1, n_out = 1, drop_nan = False):
    '''
    Converts the time-series to a supervised learning problem
    Based on: https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/

    data: Sequence of observations as a list or 2D NumPy array.
    n_in: number of lag observations as input (X). 
         Values may be between [1..len(data)] Optional. Defaults to 1.
    n_out: number of observations as output (y). 
           Values may be between [0..len(data)-1]. Optional. Defaults to 1.
    drop_nan: boolean whether or not to drop rows with NaN values. Optional. 
              (Defaults to False).
    '''
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()

    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names

    # drop rows with NaN values
    if drop_nan:
        agg.dropna(inplace=True)
    return agg

def plot_loss(history):
    
    history = history.history
    
    # Loss
    plt.plot(range(1,len(history['loss'])+1),history['loss'])
    plt.plot(range(1,len(history['val_loss'])+1),history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train set', 'Dev set'], loc='best')
    
    plt.show()

def inverse_transform(test_X, test_y, yhat, scaler):
    
    rtest_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

    # invert scaling for forecast
    inv_yhat = np.concatenate((yhat, rtest_X[:, 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]

    # invert scaling for actual
    rtest_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((rtest_y, rtest_X[:, 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]

    return inv_y, inv_yhat

def inverse_transform_multiple(test_X, test_y, yhat, scaler, n_hours, n_features):

    rtest_X = test_X.reshape((test_X.shape[0], n_hours*n_features))

    # invert scaling for forecast
    inv_yhat = np.concatenate((yhat, rtest_X[:, -7:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]

    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, rtest_X[:, -7:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]

    return inv_y, inv_yhat


def plot_comparison(series, series_label, title):
    '''
    Plot two time series, both are numpy.arrays
    '''

    plt.figure(figsize = (15, 5))

    for i in range(len(series)):
        plt.plot(series[i], label=series_label[i])
    
    plt.xlabel("x")
    plt.ylabel("Air Pollution")
    plt.title(title)
    plt.legend(loc="upper right")

    plt.show()