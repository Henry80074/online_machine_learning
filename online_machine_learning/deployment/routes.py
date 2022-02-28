import datetime
from flask import render_template, request
from deployment import app, model
import numpy as np
import pandas as pd
import pickle
from lstm_multivariate import fetch, preprocess, create_dataset
import json
import plotly
import plotly.express as px
import os
import plotly.graph_objects as go
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@app.route('/explore')
def explore():
    return render_template("search.html")


@app.route('/', methods=['GET'])
def dashboard():
    today = datetime.datetime.today()
    days = 1
    future_predict = 28
    df = fetch()
    dff = df.filter(['prices', 'value'])
    dataframe, scaler = preprocess(dff)
    # must be same as model dims
    look_back = 45
    X, Y = create_dataset(dataframe, look_back)
    Y = scaler.inverse_transform(Y)
    # make predictions
    if days == 1:
        last_batch = X[-days:]
        dates = [today + datetime.timedelta(days=x) for x in range(future_predict)]
    else:
        last_batch = X[-days:-days + 1]
        dates = [today + datetime.timedelta(days=x-days) for x in range(future_predict)]
    current = last_batch[0]
    results = []
    for i in range(future_predict):
        predict = model.predict(last_batch)
        predict_scaled = scaler.inverse_transform(predict)
        results.append(predict_scaled[0])
        current = np.append(current, predict, axis=0)
        current = np.delete(current, [0], axis=0)
        last_batch = np.array([current])
    # #get results as numpy array
    results = np.array(results)

    if days - future_predict < 0:
        fortune_teller = pd.DataFrame(
            data={'price': [col[0] for col in results], "date": dates},
            columns=["price", "date"])

        # predictions from certain day in time to X days into the future
        fig = px.line(fortune_teller, x="date", y="price", width=800, height=400,
                      title="%s days projected price" % future_predict)
    else:
        if days - future_predict == 0:
            actual = Y[-days:]

        if days - future_predict > 0:
            actual = Y[-days:-days + future_predict]

        fortune_teller = pd.DataFrame(
            data={'predictions': [col[0] for col in results], 'actual': [col[0] for col in actual]},
            columns=["predictions", "actual"])
        # predictions from certain day in time to X days into the future
        fig = px.line(fortune_teller, x=[i for i in range(len(fortune_teller))], y=fortune_teller.columns, width=800, height=400,
                      title="projected price from %s days ago to %s days ago" % (days, days - future_predict))

    # bitcoin price chart
    fig2 = px.line(df, x="date", y="prices", width=800, height=400, title="bitcoin price")
    # rolling price predictions
    pickle_in = open(ROOT_DIR + "/pickles/rolling_predictions.pkl", "rb")
    rolling_predictions_df = pickle.load(pickle_in)
    df_list, fullResults = rolling_predictions_df[0], rolling_predictions_df[1]
    # plot fig
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=fullResults.date, y=fullResults.actual_price, name="true price",
                             mode='lines'))
    fig3.add_trace(go.Scatter(x=fullResults.date, y=fullResults.predictions, mode='lines', name="predicted price",))
    for df in df_list:
        fig3.add_trace(go.Scatter(x=df_list[df].date, y=df_list[df].predictions,
                                 mode='lines'))
    fig3.update_yaxes(rangemode="nonnegative")
    fig3.update_layout(
        title_text="14 day sliding window predictions",
        autosize=False,
        width=1600,
        height=400)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    graphJSON2 = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    graphJSON3 = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template("prediction.html", graphJSON=graphJSON, graphJSON2=graphJSON2, graphJSON3=graphJSON3)


@app.route('/view_data')
def view_database():
    df = fetch()
    return render_template("data.html", tables=[df.to_html(classes='data')], titles=df.columns.values)



@app.route('/predict_one', methods=['POST'])
def predict_one():
    form_values = [int(x) for x in request.form.values()]
    days = form_values[0]
    future_predict = form_values[1]
    df = connect_and_fetch()
    df = df.filter(['prices', 'value'])
    dates = df.filter(['dates'])
    dataframe, scaler = preprocess(df)
    # must be same as model dims
    look_back = 45
    X, Y = create_dataset(dataframe, look_back)
    Y = scaler.inverse_transform(Y)
    # make predictions
    if days == 1:
        last_batch = X[-days:]
        dates = df.filter(['dates'])
    else:
        last_batch = X[-days:-days+1]
    current = last_batch[0]
    results = []
    for i in range(future_predict):
        predict = model.predict(last_batch)
        predict_scaled = scaler.inverse_transform(predict)
        results.append(predict_scaled[0])
        current = np.append(current, predict, axis=0)
        current = np.delete(current, [0], axis=0)
        last_batch = np.array([current])
    # #get results as numpy array
    results = np.array(results)

    if days - future_predict < 0:
        fortune_teller = pd.DataFrame(
            data={'predictions': [col[0] for col in results]},
            columns=["predictions"])
        # predictions from certain day in time to X days into the future
        fig = px.line(fortune_teller, x=[i for i in range(len(fortune_teller))], y="predictions",  width=800, height=400,
                      title="%s days projected price" % future_predict)
    else:
        if days - future_predict == 0:
            actual = Y[-days:]

        if days - future_predict > 0:
            actual = Y[-days:-days+future_predict]

        fortune_teller = pd.DataFrame(
            data={'predictions': [col[0] for col in results], 'actual': [col[0] for col in actual]},
            columns=["predictions", "actual"])
        # predictions from certain day in time to X days into the future
        fig = px.line(fortune_teller, x=[i for i in range(len(fortune_teller))], y=fortune_teller.columns,  width=800, height=400,
                      title="projected price from %s days ago to %s days ago" % (days, days - future_predict))

    # bitcoin price chart
    fig2 = px.line(Y, x=[i for i in range(len(Y))], y=[col[0] for col in Y], title="bitcoin price chart")
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    graphJSON2 = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template("explore.html", graphJSON=graphJSON, graphJSON2=graphJSON2)




# @app.route('/update_one', methods=['POST', 'GET'])
# def update_one():
#     # connect to fear/greed api
#     fear_greed_index = requests.get("https://api.alternative.me/fng/?limit=1")
#     # convert to dataframe
#     fear_greed_index = fear_greed_index.json()
#     print(fear_greed_index)
#     timestamp1 = [datetime.fromtimestamp((int(str(x['timestamp'])))).strftime("%d.%m.%y") for x in
#                   fear_greed_index['data']]
#     value = [x['value'] for x in fear_greed_index['data']]
#     df1 = pd.DataFrame(data={'date': [col for col in timestamp1], 'value': [col for col in value]},
#                        columns=["date", "value"])
#     # connect to coingecko api
#     bitcoin_market_data = requests.get(
#         "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=0&interval=daily")
#     # convert to dataframe
#     input_dict = bitcoin_market_data.json()
#     # get timestamps and convert to readable time, remove 3 trailing zeros to ensure correct format for conversion
#     timestamp2 = [datetime.fromtimestamp((int(str(x[0])[:-3]))).strftime("%d.%m.%y") for x in input_dict['prices']]
#     prices = [x[1] for x in input_dict['prices']]
#     market_caps = [x[1] for x in input_dict['market_caps']]
#     total_volumes = [x[1] for x in input_dict['total_volumes']]
#     df2 = pd.DataFrame(data={'date': [col for col in timestamp2], 'prices': [col for col in prices],
#                              'total_volumes': [col for col in total_volumes],
#                              'market_cap': [col for col in market_caps]},
#                        columns=["date", "prices", "total_volumes", "market_cap"])
#     # merge the dataframes
#     df3 = pd.merge(df2, df1, how="outer", on=["date"])
#     # connect to the database
#     conn = psycopg2.connect(user="postgres",
#                                   ***REMOVED***="***REMOVED***",
#                                   host="db",
#                                   port="5432",
#                                   database="online_machine_learning")
#     # create cursor
#     cursor = conn.cursor()
#     # post to database
#     # creating column list for insertion
#     cols = ",".join([str(i) for i in df3.columns.tolist()])
#     # Insert DataFrame records one by one.
#     for i, row in df3.iterrows():
#         sql = "INSERT INTO bitcoin (" + cols + ") VALUES (" + "%s," * (len(row) - 1) + "%s) ON CONFLICT ON CONSTRAINT date DO NOTHING"
#         cursor.execute(sql, tuple(row))
#         # commit to save our changes
#         conn.commit()
#     conn.close()
#     increment()
#     return render_template("data.html", tables=[df3.to_html(classes='data')], titles=df3.columns.values)
#
# @app.route('/increment', methods=['POST', 'GET'])
# def update_mod():
#     increment()
#     return "complete"
