
from flask import Flask
from tensorflow import keras
import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
model = keras.models.load_model('../') #+folder name??

from deployment import routes
