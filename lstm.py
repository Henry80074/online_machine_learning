import keras.models
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
plt.style.use('fivethirtyeight')

df = pd.read_csv('Sheet1.csv')
print(df.head())

df.to_csv(r"C:\Users\3henr\PycharmProjects\FinanceML\data.csv")

# plt.figure(figsize=(16,8))
# plt.title('price')
# plt.plot(dfnew['PRICE'])
# plt.xlabel('Date')
# plt.ylabel('price')
# plt.show()

#create new dfnew with only price - only parameter

data = df.filter(['PRICE', 'value'])

dataset = data.values
training_data_len = math.ceil(len(df)) #* 0.8)
print(training_data_len)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

#create training dataset
train_data = scaled_data[0:training_data_len, :]
#split the data into x_train and y_train data sets
x_train = []
y_train = []

n_future = 1
n_past = 14
for i in range(n_past, len(train_data)):
    x_train.append(scaled_data[i-n_past:i, 0:dataset.shape[1]])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

#x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print(x_train.shape)
print(y_train.shape)

#rmse = np.sqrt(np.mean(((predictions- y_test)**2)))
#build LSTM model
def create_model():
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(LSTM(50, return_sequences=False))
    model.add(keras.layers.Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


model = keras.models.load_model(r'C:\Users\3henr\PycharmProjects\FinanceML')
if model:
    model.summary()
else:
    model = create_model()
    # train model
    model.fit(x_train, y_train, batch_size=1, epochs=3)
    model.save(r'C:\Users\3henr\PycharmProjects\FinanceML')

n_future = 360

predictions = model.predict(x_train[-n_future:])

#get shape required for np array
prediction_copies = np.repeat(predictions, scaled_data.shape[1], axis=-1)

# get predictions into true values
trainPredict1 = scaler.inverse_transform(prediction_copies)[:,0]
print(trainPredict1)
#predictions = scaler.inverse_transform(trainPredict1.reshape(-1, 1))

#get correct index for plotting
ind = [i for i in range(training_data_len-n_future, training_data_len)]
# issue withe the index of new datafram overlapping
new_df = df.filter(["PRICE"])
new_df2 = pd.DataFrame({"index": ind , "PRICE":trainPredict1}, allow_2d=True)

# last_60_days_scaled = scaler.transform(last_60_days)
#
plt.figure(figsize=(16,8))
plt.plot(new_df2['index'], new_df2["PRICE"])
plt.plot(new_df.values)
plt.show()


# https://stackoverflow.com/questions/42997228/lstm-keras-error-valueerror-non-broadcastable-output-operand-with-shape-67704