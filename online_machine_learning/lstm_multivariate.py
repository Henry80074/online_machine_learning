import keras.models
import pandas as pd
import numpy as np
import os 
import pickle 
import shutil
import requests
from sqlalchemy import create_engine
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import ModelCheckpoint
from keras.losses import MeanSquaredError
from datetime import datetime
import psycopg2
import keras

# current directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


# connects to database, returns df
def connect_and_fetch():
    # establishing the connection
    conn = psycopg2.connect(
      database="online_machine_learning", user='postgres', ***REMOVED***='***REMOVED***', host='db', port= '5432'
    )
    sql_query = pd.read_sql_query('''
                                  SELECT
                                  *
                                  FROM bitcoin
                                  ''', conn)
    df = pd.DataFrame(sql_query, columns=['date', 'prices', 'total_volumes', 'market_cap', 'value'])
    # Closing the connection
    conn.close()
    return df


# loads model
def load():
    try:
        model = keras.models.load_model(ROOT_DIR)
        model.summary()
        return model
    except IOError:
        return print("no model found")


# returns actual and model predictions for model and dataset
def init():
    dataframe = connect_and_fetch()
    dataframe = dataframe.filter(['prices', 'value'])
    dates = dataframe.filter(['date'])
    dataframe, scaler = preprocess(dataframe)
    look_back = 45
    variables = 2
    X, Y = create_dataset(dataframe, look_back)
    model = load()
    trainPredict = model.predict(X)
    PredictScaled = scaler.inverse_transform(trainPredict)
    ActualScaled = scaler.inverse_transform(Y)
    return X, Y, PredictScaled,ActualScaled, model, scaler, dates


# creates machine learning model 
def create(look_back, variables, X_train, Y_train, X_val, Y_val):
    model = create_model(look_back, variables)
    # train model
    model_checkpoint_callback = ModelCheckpoint( # keras.callback.ModelCheckpoint changed to ModelCheckpoint??
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
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss=MeanSquaredError())
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
def plot_rolling_predictions(model, X_values, ActualScaled, PredictScaled, scaler):
    days_to_predict = 14
    predictions_list = []
    # steps through the dataframe and predicts the number of days forward, from the current batch
    for i in range(0, len(X_values), days_to_predict):
        last_batch = X_values[i:i+1]
        # removes unnecessary brackets
        current = last_batch[0]
        results = []
        # make predictions on the last batch
        for i in range(days_to_predict):
            trainPredict = model.predict(last_batch)
            trainPredictScaled = scaler.inverse_transform(trainPredict)
            results.append(trainPredictScaled[0].tolist())
            # creates np.array of batch and prediction, removes first item from array
            current = np.append(current, trainPredict, axis=0)
            current = np.delete(current, [0], axis=0)
            last_batch = np.array([current])
        # get results as numpy array
        for i in results:
            predictions_list.append(i)
    futurePredict = pd.DataFrame(predictions_list)
    n = 0
    price_column = futurePredict.iloc[:, :1]
    # fullTrainPredictScaled, fullValPredict, fullTestPredict, fullTrainActual, fullValActual, fullTestActual = init()
    # get predicted and actual price
    fullResults = pd.DataFrame(data={'predictions': [col[0] for col in PredictScaled], 'Actuals':[col[0] for col in ActualScaled]}, columns=["predictions", "Actuals"])

    data = pd.merge(fullResults, price_column, how="outer", left_index=True, right_index=True)
    today = datetime.today().strftime('%d-%m-%Y')
    try:
        os.rename(ROOT_DIR + r"\pickles\rolling_predictions.pkl",
                  ROOT_DIR + r"\pickles\rolling_predictions" + today + ".pkl")
        shutil.move(ROOT_DIR + r"\pickles\rolling_predictions" + today + ".pkl",
                    ROOT_DIR + r"\old_pickles\rollings_predictions" + today + ".pkl")
    except FileNotFoundError:
        print("file not found")
        pass
    with open(ROOT_DIR + r"\pickles\rolling_predictions.pkl", 'wb') as file:
        pickle.dump(data, file)
        print("new pickle created")
        file.close()


def get_all_data():
    fear_greed_index = requests.get("https://api.alternative.me/fng/?limit=0")
    fear_greed_index = fear_greed_index.json()
    timestamp1 = [datetime.fromtimestamp((int(str(x['timestamp'])))).strftime("%d.%m.%y") for x in fear_greed_index['data']]
    value = [x['value'] for x in fear_greed_index['data']]
    df1 = pd.DataFrame(data={'date': [col for col in timestamp1], 'value': [col for col in value]},
                      columns=["date", "value"])
    # limits amount of data from coin gecko to match fear/greed
    total_entries = len(df1)
    bitcoin_market_data = requests.get("https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=" + str(total_entries) + "&interval=daily")
    # Transform json input to python objects
    input_dict = bitcoin_market_data.json()
    # get timestamps and convert to readable time, remove 3 trailing zeros to ensure correct format for conversion
    timestamp2 = [datetime.fromtimestamp((int(str(x[0])[:-3]))).strftime("%d.%m.%y") for x in input_dict['prices']]
    prices = [x[1] for x in input_dict['prices']]
    market_caps = [x[1] for x in input_dict['market_caps']]
    total_volumes = [x[1] for x in input_dict['total_volumes']]
    df2 = pd.DataFrame(data={'date': [col for col in timestamp2], 'prices': [col for col in prices],
                                'total_volumes': [col for col in total_volumes], 'market_cap':[col for col in market_caps]},
                          columns=["date", "prices", "total_volumes", "market_cap"])
    df3 = pd.merge(df2, df1,  how="outer", on=["date"])
    # df2.set_index('date')
    df3.dropna(subset=["date", "prices", "total_volumes", "market_cap", "value"], inplace=True)
    # drops last row as this contains the current price of bitcoin
    # Drop last row
    df3.drop(index=df3.index[-1],
            axis=0,
            inplace=True)
    conn_string = "postgresql://postgres:***REMOVED***@db:5432/online_machine_learning"
    db = create_engine(conn_string)
    conn = db.connect()
    df3.to_sql('bitcoin', con=conn, if_exists='replace', index=False)


def update_rolling_predictions():
    X, Y, PredictScaled, ActualScaled, model, scaler, dates = init()
    plot_rolling_predictions(model, X, ActualScaled, PredictScaled, scaler)


def increment():
    model = keras.models.load_model(ROOT_DIR)
    df = connect_and_fetch()
    df = df.filter(['prices', 'value'])
    df, scalar = preprocess(df)
    dataX = []
    dataY = []
    # window -1, and second to last item
    X = df[-47:-2]
    dataX.append([r for r in X])
    label = df[-1]
    dataY.append(label)
    #rename model and move to old directory
    today = datetime.today().strftime('%d-%m-%Y')
    os.rename(ROOT_DIR + r"\saved_model.pb",
              ROOT_DIR + r"\saved_model" + today + ".pb")
    shutil.move(ROOT_DIR + r"\saved_model" + today  + ".pb",
                ROOT_DIR + r"\old_models\saved_model" + today  + ".pb")
    model.fit(np.array(dataX), np.array(dataY), batch_size=1, epochs=5)
    model.save(ROOT_DIR)


def update_one():
    # connect to fear/greed api
    fear_greed_index = requests.get("https://api.alternative.me/fng/?limit=1")
    # convert to dataframe
    fear_greed_index = fear_greed_index.json()
    print(fear_greed_index)
    timestamp1 = [datetime.fromtimestamp((int(str(x['timestamp'])))).strftime("%d.%m.%y") for x in
                  fear_greed_index['data']]
    value = [x['value'] for x in fear_greed_index['data']]
    df1 = pd.DataFrame(data={'date': [col for col in timestamp1], 'value': [col for col in value]},
                       columns=["date", "value"])
    # connect to coingecko api
    bitcoin_market_data = requests.get(
        "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=0&interval=daily")
    # convert to dataframe
    input_dict = bitcoin_market_data.json()
    # get timestamps and convert to readable time, remove 3 trailing zeros to ensure correct format for conversion
    timestamp2 = [datetime.fromtimestamp((int(str(x[0])[:-3]))).strftime("%d.%m.%y") for x in input_dict['prices']]
    prices = [x[1] for x in input_dict['prices']]
    market_caps = [x[1] for x in input_dict['market_caps']]
    total_volumes = [x[1] for x in input_dict['total_volumes']]
    df2 = pd.DataFrame(data={'date': [col for col in timestamp2], 'prices': [col for col in prices],
                             'total_volumes': [col for col in total_volumes],
                             'market_cap': [col for col in market_caps]},
                       columns=["date", "prices", "total_volumes", "market_cap"])
    # merge the dataframes
    df3 = pd.merge(df2, df1, how="outer", on=["date"])
    # connect to the database
    conn = psycopg2.connect(user="postgres",
                                  ***REMOVED***="***REMOVED***",
                                  host="db",
                                  port="5432",
                                  database="online_machine_learning")
    # create cursor
    cursor = conn.cursor()
    # post to database
    # creating column list for insertion
    cols = ",".join([str(i) for i in df3.columns.tolist()])
    # Insert DataFrame records one by one.
    for i, row in df3.iterrows():
        sql = "INSERT INTO bitcoin (" + cols + ") VALUES (" + "%s," * (len(row) - 1) + "%s) ON CONFLICT ON CONSTRAINT date DO NOTHING"
        cursor.execute(sql, tuple(row))
        # commit to save our changes
        conn.commit()
    conn.close()

# OLD FUNCTIONS --------------------------------

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

# #read or create pickle
# def read_or_new_pickle(path, data):
#     if os.path.isfile(path):
#         with open(path, "rb") as f:
#             try:
#                 return pickle.load(f)
#             except Exception: # so many things could go wrong, can't be more specific.
#                 pass
#     with open(path, "wb") as f:
#         pickle.dump(data, f)
#     return data
