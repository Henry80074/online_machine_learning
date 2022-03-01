import datetime
from flask import render_template, request
import numpy as np
import pandas as pd
import pickle
import json
import plotly
import plotly.express as px
import os
import plotly.graph_objects as go
from deployment import app, model
from data import fetch, preprocess, create_dataset

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
    df = fetch()
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