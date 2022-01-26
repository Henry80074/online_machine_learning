import pickle
from flask import Flask

app = Flask(__name__)
#with open(r'C:\Users\3henr\PycharmProjects\FinanceML\saved_model.pb') as file:
   # model = pickle.load(file)

from deployment import routes
