import numpy as np
import os
import shutil
import keras.models
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import ModelCheckpoint
from keras.losses import MeanSquaredError
from datetime import datetime
from sklearn.model_selection import train_test_split
from data import preprocess, fetch, create_dataset

# current directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# window
look_back = 45
# model variables
variables = 2


def init():
    df = fetch()
    dataframe = df.filter(['prices', 'value'])
    dates = df.filter(['date'])
    dataframe, scalar = preprocess(dataframe)
    x, y = create_dataset(dataframe, look_back)
    model = load()
    train_predict = model.predict(x)
    predict_scaled = scalar.inverse_transform(train_predict)
    actual_scaled = scalar.inverse_transform(y)
    return x, y, predict_scaled, actual_scaled, model, scalar, dates


def load():
    try:
        model = keras.models.load_model(ROOT_DIR)
        model.summary()
        return model
    except IOError:
        return print("no model found")


def create(look_back, variables, X_train, Y_train, X_val, Y_val):
    model = create_model(look_back, variables)
    # train model
    model_checkpoint_callback = ModelCheckpoint(
        filepath=ROOT_DIR,
        save_weights_only=True,
        monitor='val_loss',
        mode='max',
        save_best_only=True)
    model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=1, epochs=35, callbacks=[model_checkpoint_callback])
    model.save(os.getcwd())
    return model


def create_model(look_back, variables):
    model = Sequential()
    model.add(LSTM(8, return_sequences=True, input_shape=(look_back, 2)))
    model.add(LSTM(25, return_sequences=False))
    model.add(keras.layers.Dropout(0.2))
    model.add(Dense(variables))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss=MeanSquaredError())
    return model


def run():
    df = fetch()
    dataframe = df.filter(['prices', 'value'])
    dataframe, scalar = preprocess(dataframe)
    x, y = create_dataset(dataframe, look_back)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    model = create(look_back, variables, x_train, y_train, x_test, y_test)
    return model


def increment():
    model = keras.models.load_model(ROOT_DIR)
    df = fetch()
    df = df.filter(['prices', 'value'])
    df, scalar = preprocess(df)
    data_x = []
    data_y = []
    # window -1, and second to last item
    X = df[-2-look_back:-2]
    data_x.append([r for r in X])
    label = df[-1]
    data_y.append(label)
    # rename model and move to old directory
    today = datetime.today().strftime('%d-%m-%Y')
    os.rename(ROOT_DIR + "/saved_model.pb",
              ROOT_DIR + "/saved_model" + today + ".pb")
    shutil.move(ROOT_DIR + "/saved_model" + today  + ".pb",
                ROOT_DIR + "/old_models/saved_model" + today  + ".pb")
    model.fit(np.array(data_x), np.array(data_y), batch_size=1, epochs=5)
    model.save(ROOT_DIR)