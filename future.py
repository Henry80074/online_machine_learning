import os
import pickle
import shutil
from datetime import date

from lstm_multivariate import init, plot_rolling_predicitons
import matplotlib.pyplot as plt

X, Y, ActualScaled, PredictScaled, model, scaler = init()
plot_rolling_predicitons(model, X, ActualScaled, PredictScaled, scaler)

pickle_in = open("rolling_predictions", "rb")
data = pickle.load(pickle_in)
plt.plot(data)
plt.show()
