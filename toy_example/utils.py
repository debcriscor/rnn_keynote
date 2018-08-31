#
# Some functions to be used in the tutorial
#
# Developed by Debora Cristina Correa

import numpy as np
from sklearn.utils import shuffle # just shuffles the data
import pandas as pd
import matplotlib.pyplot as plt # for 2D plotting


def create_sin_data(samples=5000, period=10):
    '''
    Creates sin wave data.

    samples: number of samples
    period: length of one cycle of the curve
    '''
    # creating the sampling space
    x = np.linspace(-period * np.pi, period * np.pi, samples) 

    # Create the sin data and store it in a Dataframe format
    series = pd.DataFrame(np.sin(x))
    
    return series

def create_window(data, window_size = 50, drop_nan = False):
    '''
    Samples the data by using move window sliding

    data: sampled data in a DataFrame format
    window_size: moving window size
    drop_nan: remove the missing values (NaN)
    '''
    data_bk = data.copy()
    for i in range(window_size):
        data = pd.concat([data, data_bk.shift(-(i + 1))], 
                            axis = 1)
    
    # if drop_nan is true, we remove the NaN (Not a Number) values
    if drop_nan:
        data.dropna(axis=0, inplace=True)
        
    return data

def train_test_split( data, train_size =0.8, shuffle=True):
    '''
    data: windowed dataset in DataFrame format
    train_size: size of the training dataset
    '''

    nrow = round(train_size * data.shape[0])

    # iloc allows the using of slicing operation and returns
    # the related DataFrame. Note that, this is different of using 
    # data.values, in which the returned elements are numpy.array
    train = data.iloc[:nrow, :] # train dataset
    test = data.iloc[nrow:, :]  # test dataset

    # Shuffle training data.
    # this method shuffle arrays  in a consistent way, this reduces variance and 
    # will helps us to make sure that models remain general and overfit less, hopefully.
    if shuffle:
        train = shuffle(train)

    train_X = train.iloc[:, :-1]
    test_X = test.iloc[:, :-1]

    train_Y = train.iloc[:, -1]
    test_Y = test.iloc[:, -1]

    return train_X, train_Y, test_X, test_Y

def plot_training_test_data( data, train_size ):

    nrow = round(train_size * data.shape[0])

    # iloc allows the using of slicing operation and returns
    # the related DataFrame. Note that, this is different of using 
    # data.values, in which the returned elements are numpy.array

    f, (ax1) = plt.subplots(1,1, figsize = (15, 5))

    data.iloc[:nrow].plot(ax=ax1, color='r')
    data.iloc[nrow-1:].plot(ax=ax1, color='g')

    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Training and test data")

    ax1.legend(['Train dataset', 'Test dataset'], loc='best')

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

def plot_comparison(series_1, series_2, series_1_label,
                    series_2_label, title):
    '''
    Plot two time series, both are numpy.arrays
    '''

    plt.figure(figsize = (15, 5))
    plt.plot(series_1, label=series_1_label)
    plt.plot(series_2, label=series_2_label)
    plt.xlabel("x")
    plt.ylabel("sin(x)")
    plt.title(title)
    plt.legend(loc="upper right")

    plt.show()