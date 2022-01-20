from lstm_multivariate import get_data, preprocess, create_dataset, split, create, load
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras.models
from keras.models import Sequential
from keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

dataframe, new_entries = get_data()
dataframe = dataframe.filter(['price', 'value'])

dataframe, scaler = preprocess(dataframe)
look_back = 14
variables = 2
X, Y = create_dataset(dataframe, look_back)

model = load()
#
days = 101
last_batch  = X[-days:-days+1]
for x in last_batch:
    print(x)

current = last_batch[0]
results = []
future_predict = 100
for i in range(future_predict):
    trainPredict = model.predict(last_batch)
    trainPredictScaled = scaler.inverse_transform(trainPredict)
    results.append(trainPredictScaled[0])
    current = np.append(current, trainPredict, axis=0)
    current = np.delete(current, [0], axis=0)
    last_batch = np.array([current])

#
# #get results as numpy array
results = np.array(results)
# #get predicted and actual price
trainResults = pd.DataFrame(data={'predictions': [col[0] for col in results]}, columns=["predictions"])
#plot price_predictions
plt.plot(trainResults)
plt.show()
plt.show()



