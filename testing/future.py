import pickle
from matplotlib import pyplot as plt
from lstm_multivariate import init, plot_rolling_predictions


def update_rolling_predictions():
    X, Y, PredictScaled, ActualScaled, model, scaler, dates = init()
    plot_rolling_predictions(model, X, ActualScaled, PredictScaled, scaler)
    # pickle_in = open("rolling_predictions", "rb")
    # data = pickle.load(pickle_in)
    # plt.plot(data)
    # plt.show()
    # return data
# pickle_in = open(r"C:\Users\3henr\PycharmProjects\FinanceML\old_pickles\rollings_predictions2022-02-09.pkl", "rb")
# data = pickle.load(pickle_in)
# plt.plot(data)
# plt.show()