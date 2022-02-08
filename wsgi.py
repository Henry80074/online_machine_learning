from deployment import app
from flask_apscheduler import APScheduler
from future import update_rolling_predictions
from update import update_one, increment
import os
import keras
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def fetch_model():
    model = keras.models.load_model(ROOT_DIR)
    return model
model = fetch_model()

update_one()
increment()
update_rolling_predictions()
scheduler = APScheduler()
scheduler.init_app(app)
scheduler.add_job(id="model", func=fetch_model, trigger='interval', days=1)
scheduler.add_job(id="update_one", func=update_one, trigger='interval', days=1)
scheduler.add_job(id="increment", func=increment, trigger='interval', days=1)
scheduler.add_job(id="rolling_predict", func=update_rolling_predictions, trigger='interval', days=1)



scheduler.start()
if __name__ == '__main__':
    app.run(debug=True)