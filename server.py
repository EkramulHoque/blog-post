from flask import Flask, request, jsonify
import json
import pickle
import pandas as pd
import numpy as np
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}

api.add_resource(HelloWorld, '/')

# Load the model
model = pickle.load(open('model.pkl','rb'))
labels ={
  0: "setosa",
  1: "versicolor",
  2: "virginica"
}

@app.route('/api',methods=['POST'])
def predict():
    # Get the data from the POST request.
	data = request.get_json(force=True)
	predict = model.predict(data['feature'])
	return jsonify(labels[predict[0].tolist()])

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')


