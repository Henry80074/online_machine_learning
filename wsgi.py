from deployment import app
from flask_apscheduler import APScheduler
from update import update_one, increment

scheduler = APScheduler()
scheduler.init_app(app)
scheduler.add_job(id="update_one", func=update_one, trigger='interval', days=1)
scheduler.add_job(id="increment", func=increment, trigger='interval', days=1)
scheduler.start()
if __name__ == '__main__':
    app.run(debug=True)