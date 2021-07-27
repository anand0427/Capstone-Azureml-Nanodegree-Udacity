## %%writefile score.py
import json
import pandas as pd
import os
import pickle
import joblib
from azureml.core.model import Model

def init():
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # For multiple models, it points to the folder containing all deployed models (./azureml-models)
    model_path = Model.get_model_path('best_amlmodel_iris')
    model = joblib.load(model_path)

def run(raw_data):
    data = pd.DataFrame(json.loads(raw_data)['data'])
    # make prediction
    
    y_hat = model.predict(data)
    # you can return any data type as long as it is JSON-serializable
    return y_hat.tolist()