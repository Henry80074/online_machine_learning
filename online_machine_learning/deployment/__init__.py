from flask_sqlalchemy import SQLAlchemy
from flask import Flask
from tensorflow import keras
import os
from data import get_all_data
from lstm_model import run
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = Flask(__name__)
app.config.from_object("deployment.config.Config")
db = SQLAlchemy(app)

# get_all_data()
# run()
model = keras.models.load_model(ROOT_DIR)


class Bitcoin(db.Model):
    __tablename__ = "bitcoin"

    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.String(128), unique=True, nullable=False)
    prices = db.Column(db.Numeric(precision=8))
    total_volumes = db.Column(db.Numeric(precision=8))
    market_cap = db.Column(db.Numeric(precision=8))
    value = db.Column(db.String(128))


from deployment import routes
