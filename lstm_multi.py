import keras.models
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from connect_and_fetch import connect_and_fetch

from sklearn.metrics import mean_squared_error
plt.style.use('fivethirtyeight')

df = connect_and_fetch()
print(df.head())

df.to_csv(r"C:\Users\3henr\PycharmProjects\FinanceML\data.csv")

# plt.figure(figsize=(16,8))
# plt.title('price')
# plt.plot(df['PRICE'])
# plt.xlabel('Date')
# plt.ylabel('price')
# plt.show()

#create new df with only price - only parameter

data = df.filter(['price', 'value'])

dataset = data.values
training_data_len = math.ceil(len(df)) #* 0.8)
print(training_data_len)

# convert an array of values into a dataset matrix

def create_dataset(dataset, look_back=14):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, :])
    return np.array(dataX), np.array(dataY)

#normalize the dataset
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

# reshape into X=t and Y=t+1
look_back = 14
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
print("y_train[0:2]",trainY[0:2])
print("np.array(y_train).shape=",np.array(trainY).shape)
print('############################')

def create_model(look_back):
    model = Sequential()
    model.add(LSTM(25, return_sequences=True, input_shape=(trainX.shape[0], trainX.shape[2])))
    model.add(LSTM(25, return_sequences=False))
    model.add(keras.layers.Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

try:
    model = keras.models.load_model(r'C:\Users\3henr\PycharmProjects\FinanceML')
    model.summary()
except (IOError):
    model = create_model(look_back)
    # train model
    model.fit(trainX, trainY, batch_size=1, epochs=3)
    model.save(r'C:\Users\3henr\PycharmProjects\FinanceML')

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# Get something which has as many features as dataset
trainPredict_extended = np.zeros((len(trainPredict),2))
# Put the predictions there
trainPredict_extended[:,2] = trainPredict[:,0]
# Inverse transform it and select the 3rd column.
trainPredict = scaler.inverse_transform(trainPredict_extended) [:,1]
print(trainPredict)
# Get something which has as many features as dataset
testPredict_extended = np.zeros((len(testPredict),2))
# Put the predictions there
testPredict_extended[:,2] = testPredict[:,0]
# Inverse transform it and select the 3rd column.
testPredict = scaler.inverse_transform(testPredict_extended)[:,2]


trainY_extended = np.zeros((len(trainY),2))
trainY_extended[:,2]=trainY
trainY=scaler.inverse_transform(trainY_extended)[:,2]


testY_extended = np.zeros((len(testY),2))
testY_extended[:,2]=testY
testY=scaler.inverse_transform(testY_extended)[:,2]


# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict))
print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, 2] = trainPredict

# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, 2] = testPredict



#plot

serie,=plt.plot(scaler.inverse_transform(dataset)[:,2])
prediccion_entrenamiento,=plt.plot(trainPredictPlot[:,2],linestyle='--')
prediccion_test,=plt.plot(testPredictPlot[:,2],linestyle='--')
plt.title('Consumo de agua')
plt.ylabel('cosumo (m3)')
plt.xlabel('dia')
plt.legend([serie,prediccion_entrenamiento,prediccion_test],['serie','entrenamiento','test'], loc='upper right')