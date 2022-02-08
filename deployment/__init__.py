
from flask import Flask
from tensorflow import keras
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
print(ROOT_DIR)
app = Flask(__name__)
model = keras.models.load_model(r'C:\Users\3henr\PycharmProjects\FinanceML') #+folder name??

from deployment import routes
