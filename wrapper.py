from flask import Flask, jsonify, request
from flask_restful import API, Resource

import nlp_scprit as ns

app = Flask(__name__)
api = API(app)


@app.route('/')
def run_model(input):

    output = ns.model_obj(input)

    return jsonify(output)

if __name__ == '__main__':
    app.run()

