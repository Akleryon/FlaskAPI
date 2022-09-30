import flask
from flask import Flask, jsonify, request
import json
import pickle
from data_input import data
import numpy as np
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)


def load_models():
    file_name = "models/model_file.p"
    with open(file_name, 'rb') as pickled:
        data = pickle.load(pickled)
        model = data['model']
    return model


@app.route('/predict', methods=['GET'])


def predict():
    # parse input features from request
    request_json = request.get_json()
    x = request_json['input']
    sc = StandardScaler()
    x_in =  np.array(x).reshape(1, -1)
    x_in = sc.fit_transform(x_in)
    # load model
    model = load_models()

    prediction = model.predict(x_in)[0]
    response = json.dumps({'Prediction': str(prediction)})
    return response, 200

if __name__ == '__main__':
    application.run(debug=True)

