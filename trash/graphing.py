import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from init import init

trainPredictScaled, valPredict, testPredict, trainActual, valActual, testActual = init()

trainPredictScaled = [col[0] for col in trainPredictScaled]
trainActual = [col[0] for col in trainActual]
trainActual = np.array(trainActual).reshape(-1, 1)
trainPredictScaled = np.array(trainPredictScaled).reshape(-1, 1)
X, y = make_classification(random_state=0)

clf = SVC(random_state=0)
clf.fit(trainPredictScaled, trainActual)
SVC(random_state=0)
ConfusionMatrixDisplay.from_estimator(
     clf, testPredict, testActual)

plt.show()