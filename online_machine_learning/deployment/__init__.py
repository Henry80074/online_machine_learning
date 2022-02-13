from flask_sqlalchemy import SQLAlchemy
from flask import Flask
from tensorflow import keras
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = Flask(__name__)
app.config.from_object("deployment.config.Config")
db = SQLAlchemy(app)
model = keras.models.load_model(ROOT_DIR)

class Bitcoin(db.Model):
    __tablename__ = "bitcoin"

    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.String(128), unique=True, nullable=False)
    prices = db.Column(db.Numeric(precision=8))
    prices = db.Column(db.Numeric(precision=8))
    prices = db.Column(db.Numeric(precision=8))
    value = db.Column(db.String(128))


from deployment import routes
