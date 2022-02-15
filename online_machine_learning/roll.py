from lstm_multivariate import init
# makes prediction for multiple batch and plots it
# function to plot future predictions, requires the model, X_values (unscaled),Y values (scaled)
model, X_values, ActualScaled, PredictScaled, scaler = init()
plot_rolling_predictions()
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
plot_rolling_predictions()