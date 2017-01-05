from flask import Flask, request, jsonify
from .core import datawiz

app = Flask(__name__)


@app.route('/')
def home():
    return 'Hello, World!'


@app.route('/clean', methods=['GET','POST'])
def clean():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename == '':
            response = {"type": 'ERROR', "message": 'NO_FILE_UPLOADED'}
            return jsonify(**response)
        else:
            # TODO: Add Integrate DATAWIZ Library to process uploaded data
            # That processed data is then returned through json
            response = { 'csv_data': 'PROCESSED_DATA_PLACEHOLDER', }
            return jsonify(**response)
