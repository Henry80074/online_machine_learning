import keras.models
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from connect_and_fetch import connect_and_fetch
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError



from sklearn.metrics import mean_squared_error
plt.style.use('fivethirtyeight')

df = connect_and_fetch()
print(df.head())

df.to_csv(r"C:\Users\3henr\PycharmProjects\FinanceML\data.csv")

#create new df with only price - only parameter
dfp = pd.read_csv(r"C:\Users\3henr\PycharmProjects\FinanceML\data.csv")
price = dfp.filter(['price', 'value'])

#preprocess data to scale
scaler = StandardScaler()
scaler = scaler.fit(price)
price = scaler.transform(price)

# convert an array of values into a dataset matrix

#price = scaler.fit_transform(price)
def create_dataset(dataset, look_back):
    df_as_np = dataset
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        row = [r for r in df_as_np[i:i+look_back]]
        dataX.append(row)
        label = df_as_np[i+look_back]
        dataY.append(label)
    return np.array(dataX), np.array(dataY)

#normalize the dataset
# scaler = MinMaxScaler(feature_range=(0,1))
# scaled_data = scaler.fit_transform(dataset)

# # split into train and test sets
# training_data_len = math.ceil(len(df)) #* 0.8)
# print(training_data_len)
train_size = int(len(price) * 0.8)
val_size = int(len(price) * 0.9)
test_size = len(price)
# train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

# reshape into X=t and Y=t+1
look_back = 14
X, Y = create_dataset(price, look_back)

print(X.shape, Y.shape)
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

def create_model(look_back):
    model = Sequential()
    model.add(LSTM(25, return_sequences=True, input_shape=(look_back, 2)))
    model.add(LSTM(8, return_sequences=False))
    model.add(keras.layers.Dropout(0.2))
    model.add(Dense(2))

    model.compile(optimizer=Adam(learning_rate=0.0001), loss=MeanSquaredError())
    return model

try:
    model = keras.models.load_model(r'C:\Users\3henr\PycharmProjects\FinanceML')
    model.summary()
except (IOError):
    model = create_model(look_back)
    # train model
    model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=1, epochs=10)
    model.save(r'C:\Users\3henr\PycharmProjects\FinanceML')

# make predictions
trainPredict = model.predict(X_train)
#get shape required for np array
trainPredictScaled = scaler.inverse_transform(trainPredict)
Y_testScaled = scaler.inverse_transform(Y_train)
# get predictions into true values
results = np.array([trainPredictScaled, Y_testScaled])
#get predicted and actual price
trainResults = pd.DataFrame(data={'predictions': [col[0] for col in trainPredictScaled], 'Actuals':[col[0] for col in Y_testScaled]}, columns=["predictions", "Actuals"])
#plot price
plt.plot(trainResults)

features = 1
# # Get something which has as many features as dataset
# trainPredict_extended = np.zeros((len(trainPredict), features))
# # Put the predictions there
# trainPredict_extended[:,1] = trainPredict[:,0]
# # Inverse transform it and select the 3rd column.
# trainPredict = scaler.inverse_transform(trainPredict_extended) [:,1]
# print(trainPredict)
# # Get something which has as many features as dataset
# testPredict_extended = np.zeros((len(trainPredict),2))
# # Put the predictions there
# testPredict_extended[:,1] = trainPredict[:,0]
# # Inverse transform it and select the 3rd column.
# testPredict = scaler.inverse_transform(testPredict_extended)[:,1]
