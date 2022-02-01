import pickle
from flask import Flask
from tensorflow import keras


app = Flask(__name__)
model = keras.models.load_model('..\FinanceML')

from deployment import routes
