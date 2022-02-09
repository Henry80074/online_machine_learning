from lstm_multivariate import init
import pandas as pd
import matplotlib.pyplot as plt

# plots the data and predictions
trainPredictScaled, valPredict, testPredict, trainActual, valActual, testActual, dates = init()
#get predicted and actual price
trainResults = pd.DataFrame(data={'predictions': [col[0] for col in testPredict], 'Actuals':[col[0] for col in trainActual]}, columns=["predictions", "Actuals"])
#plot price
plt.plot(trainResults)
plt.show()



