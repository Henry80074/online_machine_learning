from deployment import app
from flask_apscheduler import APScheduler
from update import update_one, increment
from lstm_multivariate import plot_rolling_predicitons
import os
import keras
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
scheduler = APScheduler()
scheduler.init_app(app)
scheduler.add_job(id="update_one", func=update_one, trigger='interval', days=1)
scheduler.add_job(id="increment", func=increment, trigger='interval', days=1)
scheduler.add_job(id="rolling_predict", func=plot_rolling_predicitons, trigger='interval', days=1)

def fetch_model():
    model = keras.models.load_model(ROOT_DIR) 
    return model
    
model = fetch_model()

scheduler.start()
if __name__ == '__main__':
    app.run(debug=True)