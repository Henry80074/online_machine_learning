import keras.models
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from connect_and_fetch1 import connect_and_fetch
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from sklearn.metrics import mean_squared_error

plt.style.use('fivethirtyeight')

def get_data():
    dfnew = connect_and_fetch()
    dfnew.to_csv(r"C:\Users\3henr\PycharmProjects\FinanceML\datanew.csv")
    # create new dfnew with only price - only parameter
    dfold = pd.read_csv(r"C:\Users\3henr\PycharmProjects\FinanceML\dataold.csv")
    dataframe = dfold.filter(['price', 'value'])
    no_entries = len(dfnew) - len(dfold)
    if no_entries > 0:
        new_entries = dfnew[len(dfold) + 1:]  # gets new entries #to do: index needs shifting 1 back
    return dataframe, new_entries


def load_or_create():
    try:
        model = keras.models.load_model(r'C:\Users\3henr\PycharmProjects\FinanceML')
        model.summary()
        return model
    except IOError:
        model = create_model(look_back, variables)
        # train model
        model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=1, epochs=10)
        model.save(r'C:\Users\3henr\PycharmProjects\FinanceML')
        return model


def create_model(look_back, variables):
    model = Sequential()
    model.add(LSTM(25, return_sequences=True, input_shape=(look_back, 2)))
    model.add(LSTM(8, return_sequences=False))
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


# split into train, validation and test sets
def split(dataset):
    train_size = int(len(dataset) * 0.8)
    val_size = int(len(dataset) * 0.9)
    test_size = len(dataset)
    return train_size, val_size, test_size


dataframe, new_entries = get_data()
dataframe = dataframe.filter(['price', 'value'])

dataframe, scaler = preprocess(dataframe)
look_back = 14
variables = 2
X, Y = create_dataset(dataframe, look_back)

print(X.shape, Y.shape)
train_size, val_size, test_size = split(dataframe)
X_train, Y_train = X[:train_size], Y[:train_size]
X_val, Y_val = X[train_size: val_size], Y[train_size: val_size]
X_test, Y_test = X[val_size:test_size], Y[val_size:test_size]
print("np.array(y_train).shape=",Y_train.shape)
print("np.array(y_val).shape=",Y_val.shape)
print("np.array(y_test).shape=",Y_test.shape)
print("np.array(x_train).shape=",X_train.shape)
print("np.array(x_val).shape=",X_val.shape)
print("np.array(x_test).shape=",X_test.shape)
print('############################')


model = load_or_create()
# make predictions
trainPredict = model.predict(X_test)

# get predictions into true values
trainPredictScaled = scaler.inverse_transform(trainPredict)
Y_testScaled = scaler.inverse_transform(Y_test)
#get shape required for np array
results = np.array([trainPredictScaled, Y_testScaled])
#get predicted and actual price
trainResults = pd.DataFrame(data={'predictions': [col[0] for col in trainPredictScaled], 'Actuals':[col[0] for col in Y_testScaled]}, columns=["predictions", "Actuals"])
#plot price
plt.plot(trainResults)
plt.show()