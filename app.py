# importing libraries
from flask import Flask , jsonify , request
import pandas as pd
import numpy as np
from sklearn.externals import joblib
import json

# ML
from sklearn.preprocessing import StandardScaler,LabelEncoder,OnehotEncoder
from sklearn.compose import ColumnTransformer

app = Flask(__name__)


@app.route("/")
def index():
    return flask.render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():




if __name__ =="__main__":
    app.run(host="0.0.0.0" , port=5000)    
