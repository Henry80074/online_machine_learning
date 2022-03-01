import numpy as np
import pandas as pd
import psycopg2
import requests
from sklearn.preprocessing import StandardScaler
from datetime import datetime


def connect():
    conn = psycopg2.connect(
        database="online_machine_learning",
        user='postgres',
        ***REMOVED***='***REMOVED***',
        host='localhost',
        port='5432')
    return conn


def fetch():
    # establishing the connection
    conn = connect()
    sql_query = pd.read_sql_query('''SELECT * FROM bitcoin''', conn)
    df = pd.DataFrame(sql_query, columns=['date', 'prices', 'total_volumes', 'market_cap', 'value'])
    # Closing the connection
    conn.close()
    return df


# preprocess data to scale
def preprocess(dataset):
    scaler = StandardScaler()
    scaler = scaler.fit(dataset)
    scaled_dataset = scaler.transform(dataset)
    return scaled_dataset, scaler


# convert an array of values into a dataset matrix
def create_dataset(dataset, window):
    df_as_np = dataset
    data_x, data_y = [], []
    for i in range(len(dataset) - window):
        row = [r for r in df_as_np[i:i+window]]
        data_x.append(row)
        label = df_as_np[i+window]
        data_y.append(label)
    return np.array(data_x), np.array(data_y)


# sends request to apis and gets data into dataframe
def set_up_api(limit):
    fear_greed_index = requests.get("https://api.alternative.me/fng/?limit=" + str(limit))
    fear_greed_index = fear_greed_index.json()
    timestamps = [datetime.fromtimestamp((int(str(x['timestamp'])))).strftime("%d.%m.%y") for x in
                  fear_greed_index['data']]
    value = [x['value'] for x in fear_greed_index['data']]
    df1 = pd.DataFrame(data={'date': [col for col in timestamps], 'value': [col for col in value]},
                       columns=["date", "value"])
    # limits amount of data from coin gecko to match fear/greed
    total_entries = len(df1)
    bitcoin_market_data = requests.get(
        "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=" + str(
            total_entries) + "&interval=daily")
    # Transform json input to python objects
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
    df3 = pd.merge(df2, df1, how="outer", on=["date"])
    return df3


def insert_data(df3):
    conn = connect()
    # create cursor
    cursor = conn.cursor()
    # post to database
    # creating column list for insertion
    cols = ",".join([str(i) for i in df3.columns.tolist()])
    # Insert DataFrame records one by one.
    for i, row in df3.iterrows():
        sql = "INSERT INTO bitcoin (" + cols + ") VALUES (" + "%s," * (len(row) - 1) + "%s) ON CONFLICT (date) DO NOTHING"
        cursor.execute(sql, tuple(row))
        # commit to save our changes
        conn.commit()
    conn.close()


def update_one():
    # collects last data entries
    df3 = set_up_api(limit=1)
    # insert data
    insert_data(df3)


def get_all_data():
    # limit zero gets all past data
    df3 = set_up_api(limit=0)
    # removes na values
    df3.dropna(subset=["date", "prices", "total_volumes", "market_cap", "value"], inplace=True)
    # drops last row as this contains the current price of bitcoin
    df3.drop(index=df3.index[-1], axis=0, inplace=True)
    insert_data(df3)
