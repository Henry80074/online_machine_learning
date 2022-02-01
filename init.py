from lstm_multivariate import get_data, preprocess, create_dataset, split, create, load
import pandas as pd
import matplotlib.pyplot as plt


def init():
    dataframe, new_entries = get_data()
    dataframe = dataframe.filter(['prices', 'value'])
    dataframe, scaler = preprocess(dataframe)
    look_back = 45
    variables = 2
    X, Y = create_dataset(dataframe, look_back)

    #split data, separate.
    train_size, val_size, test_size = split(dataframe)
    X_train, Y_train = X[:train_size], Y[:train_size]
    X_val, Y_val = X[train_size: val_size], Y[train_size: val_size]
    X_test, Y_test = X[val_size:test_size], Y[val_size:test_size]

    model = load()
    #model = create(look_back, variables, X_train, Y_train, X_val, Y_val)
    # model_checkpoint_callback = tensorflow.keras.callbacks.ModelCheckpoint(
    #     filepath="/tmp/checkpoint",
    #     save_weights_only=True,
    #     monitor='val_accuracy',
    #     mode='max',
    #     save_best_only=True)

    # Model weights are saved at the end of every epoch, if it's the best seen
    # so far.
    # model.fit(epochs=EPOCHS, callbacks=[model_checkpoint_callback])
    # make predictions
    trainPredict = model.predict(X)
    # valPredict = model.predict(X_val)
    # testPredict = model.predict(X_test)
    # get predictions into true values
    PredictScaled = scaler.inverse_transform(trainPredict)
    ActualScaled = scaler.inverse_transform(Y)
    # valActual = scalar.inverse_transform(Y_val)
    # testActual = scalar.inverse_transform(Y_test)


    #get results as numpy array
    #results = np.array([trainPredict, trainActual])
    return X, Y, PredictScaled,ActualScaled, model, scaler

# plots the data and predictions
trainPredictScaled, valPredict, testPredict, trainActual, valActual, testActual = init()
#get predicted and actual price
trainResults = pd.DataFrame(data={'predictions': [col[0] for col in testPredict], 'Actuals':[col[0] for col in trainActual]}, columns=["predictions", "Actuals"])
#plot price
plt.plot(trainResults)
plt.show()
plt.show()



