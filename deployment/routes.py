import psycopg2
from flask import flash, render_template, url_for, request, redirect, jsonify
from deployment import app, model
import numpy as np
import matplotlib.pyplot as plt, mpld3
import pandas as pd
import pickle
from lstm_multivariate import connect_and_fetch, preprocess, create_dataset
import json
import plotly
import plotly.express as px
import plotly.graph_objects as go

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Sales should be $ {}'.format(output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

@app.route('/dashboard')
def dashboard():
    return "dashboard"

@app.route('/view_data')
def view_database():
    df = connect_and_fetch()
    return render_template("data_table.html", tables=[df.to_html(classes='data')], titles=df.columns.values)

@app.route('/predict_one')
def predict_one():
    df = connect_and_fetch()
    df = df.filter(['price', 'value'])
    dataframe, scaler = preprocess(df)
    # must be same as model dims
    look_back = 45
    X, Y = create_dataset(dataframe, look_back)
    Y = scaler.inverse_transform(Y)
    # make predictions
    days = 14
    if days == 1:
        last_batch = X[-days:]
    else:
        last_batch = X[-days:-days+1]
    current = last_batch[0]
    results = []
    future_predict = 14
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
        fig = px.line(fortune_teller, x=[i for i in range(len(fortune_teller))], y="predictions",
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
        fig = px.line(fortune_teller, x=[i for i in range(len(fortune_teller))], y=fortune_teller.columns,
                      title="projected price from %s days ago to %s days ago" % (days, days - future_predict))



    # bitcoin price chart
    fig2 = px.line(Y, x=[i for i in range(len(Y))], y=[col[0] for col in Y])


    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    graphJSON2 = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template("data_table.html", tables=[fortune_teller.to_html(classes='data')], titles=df.columns.values, graphJSON=graphJSON, graphJSON2=graphJSON2)