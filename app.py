from keras.models import Sequential
from keras.layers import CuDNNLSTM, BatchNormalization, Dense, Dropout
import pandas as pd
import numpy as np


df = pd.read_csv("FILEPATH.csv",
                 parse_dates=True, infer_datetime_format=True, keep_default_na=False, keep_date_col=True,
                 low_memory=False, index_col=[0])


def create_datasets(df):
    training_size = int(0.7*len(df))
    train_x, train_y = [], []
    test_x, test_y = [], []
    counter = 0
    for timestamps in df:
        if counter < training_size:
            train_x.append(timestamps - df.columns['Close'])
            train_y.append(df.columns['Close'])
        else:
            test_x.append(timestamps - df.columns['Close'])
            test_y.append(df.columns['Close'])
    return train_x, train_y, test_x, test_y


print(df.columns)
print(df.shape)


def work_your_magic_df(train_x, train_y, test_x, test_y):
    train_x, train_y, test_x, test_y = create_datasets(df)
    train_x = np.reshape(train_x, len(train_x), 1)
    train_y = np.reshape(train_y, len(train_y), 1)
    test_x = np.reshape(test_x, len(test_x), 1)
    test_y = np.reshape(test_y, len(test_y), 1)

    model = Sequential()
    model.add(CuDNNLSTM(128, input_shape=(train_x.shape[1:])))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    # end of first layer
    model.add(CuDNNLSTM(256))
    model.add(Dropout(0.15))
    model.add(BatchNormalization())
    # end of second layer
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(7, activation='softmax'))
    # end of neural network

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    model.fit(train_x, train_y, epochs=25, batch_size=5)
