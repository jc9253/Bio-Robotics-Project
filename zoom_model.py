%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import queue

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Embedding, LSTM
from tensorflow.keras.preprocessing import sequence, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
from keras.utils import np_utils
from IPython.display import SVG


class zoom_model():
    def __init__(self, batch_size, time_steps, series_shape):

        self.batch_size = batch_size

        # LSTM Model Definition
        self.model = Sequential()
        self.model.add(LSTM(100, batch_input_shape=(self.batch_size, time_steps, 
                                                    series_shape), 
                                                    dropout=0.0, 
                                                    recurrent_dropout=0.0))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(20,activation='relu'))
        self.model.add(Dense(1,activation='linear'))
        self.model.compile(loss='mean_squared_error', 
                        optimizer='sgd',
                        metrics=['mae'])

        pass
    
    def make_training(self, BATCH_SIZE):
        self.data_train = 
        self.labels_train = 

        self.data_val = 
        self.labels_val = 
        pass

    def train(self, epochs):
        self.history = self.model.fit(
            self.data_train,
            self.labels_train,
            batch_size=self.batch_size,
            epochs=epochs,
            verbose=1,
            validation_data= (
                self.data_val, 
                self.labels_val, 
            ),
        )

        plt.figure()
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

        score = self.lstm_model.evaluate(self.data_train, self.labels_train, verbose=1)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])


    def infer(self, time_series):
        return self.model.predict(time_series)
