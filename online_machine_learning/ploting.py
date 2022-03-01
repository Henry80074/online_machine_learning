import pandas as pd
import numpy as np
import os
import pickle
import shutil
from datetime import datetime
from data import fetch
from lstm_model import init

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


# makes prediction for multiple batch and plots it
# function to plot future predictions, requires the model, x_values (unscaled),Y values (scaled)
def plot_rolling_predictions(model, x_values, actual_scaled, predict_scaled, scaler):
    days_to_predict = 14
    predictions_list = []
    df = fetch()
    lstm_window = 45
    df = df.filter(['prices', 'value', 'date']).loc[lstm_window:]
    dates_list = []
    date_column = df.iloc[:, 2:]
    for i in range(0, len(date_column), days_to_predict):
        dates = date_column[i:i+days_to_predict]
        dates_list.append(dates.values.tolist())

    # steps through the dataframe and predicts the number of days forward, from the current batch
    for i in range(0, len(x_values), days_to_predict):
        last_batch = x_values[i:i + 1]
        # removes unnecessary brackets
        current = last_batch[0]
        results = []
        # make predictions on the last batch
        for i in range(days_to_predict):
            train_predict = model.predict(last_batch)
            train_predict_scaled = scaler.inverse_transform(train_predict)
            results.append(train_predict_scaled[0].tolist())
            # creates np.array of batch and prediction, removes first item from array
            current = np.append(current, train_predict, axis=0)
            current = np.delete(current, [0], axis=0)
            last_batch = np.array([current])
        # get results as numpy array
        predictions_list.append(results)
    for i in range(len(dates_list)):
        for num in range(len(predictions_list[i])):
            try:
                predictions_list[i][num].append(dates_list[i][num][0])
            except IndexError:
                pass
    df_list = {}
    for i in range(len(predictions_list)):
        batch = predictions_list[i]
        try:
            df = pd.DataFrame(data={'predictions': [col[0] for col in batch], 'date':[col[2] for col in batch]}, columns=["predictions", "date"])
            df_list[i] = df
        except IndexError:
            pass
    # get predicted and actual price
    print("predict" + str(len(predict_scaled)))
    print("actual" + str(len(actual_scaled)))
    print("date" + str(len(date_column.values.tolist())))
    full_results = pd.DataFrame(data={'predictions': [col[0] for col in predict_scaled], 'actual_price':[col[0] for col in actual_scaled], "date": [x[0] for x in date_column.values.tolist()]}, columns=["predictions", "actual_price", "date"])
    data = [df_list, full_results]

    today = datetime.today().strftime('%d-%m-%Y')
    try:
        os.rename(ROOT_DIR + "/pickles/rolling_predictions.pkl",
                  ROOT_DIR + "/pickles/rolling_predictions" + today + ".pkl")
        shutil.move(ROOT_DIR + "/pickles/rolling_predictions" + today + ".pkl",
                    ROOT_DIR + "/old_pickles/rollings_predictions" + today + ".pkl")
    except FileNotFoundError:
        print("file not found")
        pass
    with open(ROOT_DIR + "/pickles/rolling_predictions.pkl", 'wb') as file:
        pickle.dump(data, file)
        print("new pickle created")
        file.close()


def update_rolling_predictions():
    x, y, predict_scaled, actual_scaled, model, scaler, dates = init()
    plot_rolling_predictions(model, x, actual_scaled, predict_scaled, scaler)





