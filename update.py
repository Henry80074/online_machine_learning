from datetime import datetime
import psycopg2
from deployment import model
import numpy as np
import pandas as pd
from lstm_multivariate import connect_and_fetch, preprocess
import requests


def increment():
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
    model.fit(np.array(dataX), np.array(dataY), batch_size=1, epochs=5)
    model.save(r'C:\Users\3henr\PycharmProjects\FinanceML')


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
                                  host="127.0.0.1",
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
    print("database updated: " + df3)