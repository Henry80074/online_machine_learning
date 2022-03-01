from keras.optimizer_v2.adam import Adam
from tensorflow import keras
import keras.models
import keras.losses
import keras.optimizers
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.losses import MeanSquaredError
from ploting import connect_and_fetch, preprocess, create_dataset


def create_model(look_back, variables):
    model = Sequential()
    model.add(LSTM(8, return_sequences=True, input_shape=(look_back, variables)))
    model.add(LSTM(25, return_sequences=False))
    model.add(keras.layers.Dropout(0.2))
    model.add(Dense(variables))
    model.compile(optimizer=Adam(learning_rate=0.0001), loss=MeanSquaredError(), metrics="acc")
    return model


def create(look_back, variables, X_train, Y_train, X_val, Y_val):
    model = create_model(look_back, variables)
    # train model
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=(r'C:\Users\3henr\PycharmProjects\FinanceML'),
        save_weights_only=True,
        monitor='val_loss',
        mode='max',
        save_best_only=True)
    model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=1, epochs=35, callbacks=[model_checkpoint_callback])
    model.save(r'C:\Users\3henr\PycharmProjects\FinanceML')
    return model

def run():
    look_back = 45
    variables = 2
    df = connect_and_fetch()
    dataframe = df.filter(['prices', 'value'])
    dataframe, scalar = preprocess(dataframe)
    X, Y = create_dataset(dataframe, look_back)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
    model = create(look_back, variables, X_train, Y_train, X_test, Y_test)
    return model
