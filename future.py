from lstm_multivariate import get_data, preprocess, create_dataset, split, create, load
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataframe, new_entries = get_data()
dataframe = dataframe.filter(['price', 'value'])

dataframe, scaler = preprocess(dataframe)
look_back = 90
future = 7
variables = 2
X, Y = create_dataset(dataframe, look_back)

print(X.shape, Y.shape)

#split data, separate.
train_size, val_size, test_size = split(dataframe)
X_train, Y_train = X[:train_size], Y[:train_size]
X_val, Y_val = X[train_size: val_size], Y[train_size: val_size]
X_test, Y_test = X[val_size:test_size], Y[val_size:test_size]

#shapes of data
print("np.array(y_train).shape=",Y_train.shape)
print("np.array(y_val).shape=",Y_val.shape)
print("np.array(y_test).shape=",Y_test.shape)
print("np.array(x_train).shape=",X_train.shape)
print("np.array(x_val).shape=",X_val.shape)
print("np.array(x_test).shape=",X_test.shape)
print('############################')

model = load()
#model = create(look_back, variables, X_train, Y_train, X_val, Y_val)
# make predictions
trainPredict = model.predict(X)
valPredict = model.predict(X_val)
testPredict = model.predict(X_test)
# get predictions into true values
trainPredictScaled = scaler.inverse_transform(trainPredict)
trainActual = scaler.inverse_transform(Y)
valActual = scaler.inverse_transform(Y_val)
testActual = scaler.inverse_transform(Y_test)


#get results as numpy array
results = np.array([trainPredict, trainActual])
#get predicted and actual price
trainResults = pd.DataFrame(data={'predictions': [col[0] for col in trainPredictScaled], 'Actuals':[col[0] for col in trainActual]}, columns=["predictions", "Actuals"])
#plot price
plt.plot(trainResults)
plt.show()
plt.show()



