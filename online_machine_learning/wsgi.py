from deployment import app, db
from flask.cli import FlaskGroup
from flask_apscheduler import APScheduler
from data import update_one, get_all_data
from ploting import update_rolling_predictions
from lstm_model import increment
import os
import keras

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def fetch_model():
    return keras.models.load_model(ROOT_DIR)


cli = FlaskGroup(app)


@cli.command("create_db")
def create_db():
    db.drop_all()
    db.create_all()
    db.session.commit()


@cli.command("get_data")
def get_data():
    get_all_data()


@cli.command("update_one")
def update_1():
    update_one()


@cli.command("increment")
def train_one():
    increment()


@cli.command("update_rolling_predictions")
def update_predict():
    update_rolling_predictions()


model = fetch_model()

scheduler = APScheduler()
scheduler.init_app(app)
scheduler.add_job(id="model", func=fetch_model, trigger='interval', days=1)
scheduler.add_job(id="update_one", func=update_one, trigger='interval', days=1)
scheduler.add_job(id="run", func=increment, trigger='interval', days=14)
scheduler.add_job(id="rolling_predict", func=update_rolling_predictions, trigger='interval', days=1)
scheduler.start()

if __name__ == '__main__':
    cli()
