import keras.models
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os 
import pickle 
import shutil
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
import psycopg2
from datetime import date
plt.style.use('fivethirtyeight')

#current directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# connects to database, returns df
def connect_and_fetch():
   #establishing the connection
   conn = psycopg2.connect(
      database="online_machine_learning", user='postgres', ***REMOVED***='***REMOVED***', host='127.0.0.1', port= '5432'
   )
   sql_query = pd.read_sql_query('''
                                  SELECT
                                  *
                                  FROM bitcoin
                                  ''', conn)

   df = pd.DataFrame(sql_query, columns=['date', 'prices', 'total_volumes', 'market_cap', 'value'])
   #Closing the connection
   conn.close()
   return df

def load():
    try:
        model = keras.models.load_model(ROOT_DIR)
        model.summary()
        return model
    except IOError:
        return print(("no model found"))

# returns actual and model predictions for model and dataset
def init():
    dataframe = connect_and_fetch()
    dataframe = dataframe.filter(['prices', 'value'])
    dataframe, scaler = preprocess(dataframe)
    look_back = 45
    variables = 2
    X, Y = create_dataset(dataframe, look_back)
    model = load()
    trainPredict = model.predict(X)
    PredictScaled = scaler.inverse_transform(trainPredict)
    ActualScaled = scaler.inverse_transform(Y)
    return X, Y, PredictScaled,ActualScaled, model, scaler

# creates machine learning model 
def create(look_back, variables, X_train, Y_train, X_val, Y_val):
        model = create_model(look_back, variables)
        # train model
        model_checkpoint_callback = ModelCheckpoint( #keras.callback.ModelCheckpoint changed to ModelCheckpoint??
            filepath=ROOT_DIR,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)
        model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=1, epochs=35, callbacks=[model_checkpoint_callback])
        model.save(os.getcwd)
        return model

# model topology
def create_model(look_back, variables):
    model = Sequential()
    model.add(LSTM(8, return_sequences=True, input_shape=(look_back, 2)))
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
    today = str(date.today())
    os.rename(ROOT_DIR + r"\rolling_predictions",
              ROOT_DIR + r"\rolling_predictions" + today)
    shutil.move(ROOT_DIR + r"\rolling_predictions" + today,
                ROOT_DIR + r"\old_pickles\rollings_predictions" + today)
    pickle_out = open("/rolling_predictions", "wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()

# def get_x_y(all_data, new_entries, window):
#     dataX = []
#     dataY = []
#     for i in range(new_entries):
#         X = all_data[-new_entries - window + i: -new_entries + i]
#         dataX.append([r for r in X])
#         label = all_data[-new_entries + i]
#         dataY.append(label)
#     return np.array(dataX), np.array(dataY)

# split into train, validation and test sets
# def split(dataset):
#     train_size = int(len(dataset) * 0.8)
#     val_size = int(len(dataset) * 0.9)
#     test_size = len(dataset)
#     return train_size, val_size, test_size
