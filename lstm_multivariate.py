import keras.models
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from sklearn.metrics import mean_squared_error
import psycopg2
plt.style.use('fivethirtyeight')


def connect_and_fetch():

   #establishing the connection
   conn = psycopg2.connect(
      database="fear_and_greed", user='postgres', ***REMOVED***='224822', host='127.0.0.1', port= '5432'
   )
   sql_query = pd.read_sql_query('''
                                  SELECT
                                  *
                                  FROM fear_greed_index
                                  ORDER BY timestamp''', conn)

   df = pd.DataFrame(sql_query, columns=['timestamp', 'value', 'price', 'value_classification'])

   #Closing the connection
   conn.close()
   return df


def get_data():
    dfnew = connect_and_fetch()
    dfnew.to_csv(r"C:\Users\3henr\PycharmProjects\FinanceML\datanew.csv")
    # create new dfnew with only price - only parameter
    try:
        dfold = pd.read_csv(r"C:\Users\3henr\PycharmProjects\FinanceML\dataold.csv")
        no_entries = len(dfnew) - len(dfold)
        if no_entries > 0:
            new_entries = dfnew[len(dfold) + 1:]  # gets new entries #to do: index needs shifting 1 back
    except:
        new_entries = None
    return dfnew, new_entries

def load():
    try:
        model = keras.models.load_model(r'C:\Users\3henr\PycharmProjects\FinanceML')
        model.summary()
        return model
    except IOError:
        return print(("no model found"))

def create(look_back, variables, X_train, Y_train, X_val, Y_val):
        model = create_model(look_back, variables)
        # train model
        model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=1, epochs=50)
        model.save(r'C:\Users\3henr\PycharmProjects\FinanceML')
        return model


def create_model(look_back, variables):
    model = Sequential()
    model.add(LSTM(25, return_sequences=True, input_shape=(look_back, 2)))
    model.add(LSTM(25, return_sequences=False))
    model.add(keras.layers.Dropout(0.2))
    model.add(Dense(variables))
    model.compile(optimizer=Adam(learning_rate=0.0001), loss=MeanSquaredError())
    return model


# preprocess data to scale
def preprocess(dataset):
    scaler = StandardScaler()
    scaler = scaler.fit(dataset)
    scaled_dataset = scaler.transform(dataset)
    return scaled_dataset, scaler


# convert an array of values into a dataset matrix
def create_dataset(dataset, window):
    df_as_np = dataset
    dataX, dataY = [], []
    for i in range(len(dataset) - window):
        row = [r for r in df_as_np[i:i+window]]
        dataX.append(row)
        label = df_as_np[i+window]
        dataY.append(label)
    return np.array(dataX), np.array(dataY)


def get_x_y(all_data, new_entries, window):
    dataX = []
    dataY = []
    for i in range(new_entries):
        X = all_data[-new_entries - window + i: -new_entries + i]
        dataX.append([r for r in X])
        label = all_data[-new_entries + i]
        dataY.append(label)
    return np.array(dataX), np.array(dataY)

# split into train, validation and test sets
def split(dataset):
    train_size = int(len(dataset) * 0.8)
    val_size = int(len(dataset) * 0.9)
    test_size = len(dataset)
    return train_size, val_size, test_size






