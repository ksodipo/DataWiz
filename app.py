from flask import Flask
from .core import datawiz
app = Flask(__name__)


@app.route('/')
def home():
    return 'Hello, World!'


@app.route('/clean')
def clean():
    # Use datawiz library to process data
    response = { 'csv_data': d, }
    pass
