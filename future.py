import datetime

import requests

from init import init
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# fear_greed_index = requests.get("https://api.alternative.me/fng/?limit=0")
#     #print(response.json())
# bitcoin_market_data = requests.get("https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=1365&interval=daily")
# # Transform json input to python objects
# input_dict = bitcoin_market_data.json()
# # Filter python objects with list comprehensions
# prices = [x for x in input_dict['prices']]
# print(prices)
# market_caps = [x for x in input_dict['market_caps']]
# total_volumes = [x for x in input_dict['total_volumes']]
# df = pd.DataFrame(data={'timestamp': [int(col[0]) for col in prices], 'prices': [col[1] for col in prices],
#                             'total_volumes': [col[1] for col in total_volumes], 'market_cap':[col[1] for col in market_caps]},
#                       columns=["timestamp", "prices", "total_volumes", "market_cap"])


#*********************************************
# #makes prediction for 1 batch and plots it
# days = 14
# last_batch = X[-days:-days+1]
# for x in last_batch:
#     print(x)
# print(scaler.inverse_transform(last_batch[0]))
# current = last_batch[0]
# results = []
# future_predict = 14
# for i in range(future_predict):
#     trainPredict = model.predict(last_batch)
#     trainPredictScaled = scaler.inverse_transform(trainPredict)
#     results.append(trainPredictScaled[0])
#     current = np.append(current, trainPredict, axis=0)
#     current = np.delete(current, [0], axis=0)
#     last_batch = np.array([current])
#
# # #get results as numpy array
# results = np.array(results)
#
#
# #
# # #get predicted and actual price
# futurePredict = pd.DataFrame(data={'predictions_y': [col[0] for col in results]}, columns=["predictions_y"], index=[i for i in range(1352,1366)])
# #plot price_predictions
# predictArray = np.empty(shape=(len(dataframe), 1))
# df = pd.DataFrame(data=predictArray, columns=["price"])
# fullTrainPredictScaled, fullValPredict, fullTestPredict, fullTrainActual, fullValActual, fullTestActual = init()
# #get predicted and actual price
# fullResults = pd.DataFrame(data={'predictions': [col[0] for col in fullTrainPredictScaled], 'Actuals':[col[0] for col in fullTrainActual]}, columns=["predictions", "Actuals"])
# #plot price
#
# new = pd.merge(fullResults, futurePredict, how="outer", left_index=True, right_index=True)
#
# plt.plot(new)
#plt.show()
#*********************************************

#consider pickling this object to load again with ease/??

# makes prediction for multiple batch and plots it
# function to plot future predictions, requires the model, X_values (unscaled),Y values (scaled)
def plot_rolling_predicitons(model, X_values, ActualScaled, PredictScaled, scaler):
    days_to_predict = 14
    predictions_list = []
    #steps through the dataframe and predicts the number of days forward, from the current batch
    for i in range(0, len(X_values), days_to_predict):
        last_batch = X_values[i:i+1]
        #removes unnecessary brackets
        current = last_batch[0]
        results = []
        #make predictions on the last batch
        for i in range(days_to_predict):
            trainPredict = model.predict(last_batch)
            trainPredictScaled = scaler.inverse_transform(trainPredict)
            results.append(trainPredictScaled[0].tolist())
            # creates np.array of batch and prediction, removes first item from array
            current = np.append(current, trainPredict, axis=0)
            current = np.delete(current, [0], axis=0)
            last_batch = np.array([current])
        #get results as numpy array
        for i in results:
            predictions_list.append(i)
    futurePredict = pd.DataFrame(predictions_list)
    n = 0
    price_column = futurePredict.iloc[: , :1]

    #fullTrainPredictScaled, fullValPredict, fullTestPredict, fullTrainActual, fullValActual, fullTestActual = init()
    #get predicted and actual price
    fullResults = pd.DataFrame(data={'predictions': [col[0] for col in PredictScaled], 'Actuals':[col[0] for col in ActualScaled]}, columns=["predictions", "Actuals"])

    data = pd.merge(fullResults, price_column, how="outer", left_index=True, right_index=True)
    plt.plot(data)
    plt.show()

X, Y, ActualScaled, PredictScaled, model, scaler= init()

plot_rolling_predicitons(model, X, ActualScaled, PredictScaled, scaler)
plt.show()