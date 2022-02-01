from lstm_multivariate import get_data, preprocess, create_dataset, split, load, get_x_y
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

all_data, new_entries = get_data()
dataframe = all_data.filter(['price', 'value'])
#
dataframe, scaler = preprocess(dataframe)
x, y = get_x_y(dataframe, len(new_entries), 14)

print(x[:].shape)
print(y[:].shape)

# dataframe, scalar = preprocess(dataframe)
look_back = 14
variables = 2

model = load()
# # make predictions
#model.train_on_batch(x, y)
# # predict from new batch
trainPredict = model.predict_on_batch(x[:])
Y_testScaled = scaler.inverse_transform(y[:])
# get predictions into true values
trainPredictScaled = scaler.inverse_transform(trainPredict)
Y_testScaled = scaler.inverse_transform(y)


#get shape required for np array
results = np.array([trainPredictScaled, Y_testScaled])
#get predicted and actual price
trainResults = pd.DataFrame(data={'predictions': [col[0] for col in trainPredictScaled], 'Actuals':[col[0] for col in Y_testScaled]}, columns=["predictions", "Actuals"])
#plot price
plt.plot(trainResults)

plt.show()