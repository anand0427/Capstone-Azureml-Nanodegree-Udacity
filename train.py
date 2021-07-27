from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.core.dataset import Dataset

run = Run.get_context()

def clean_data(data):
    # Dict for cleaning data
    
    y_df = data["Column5"]
    x_df = data.drop("Column5", axis=1)
    
    
    return x_df, y_df

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", float(args.C))
    run.log("Max iterations:", int(args.max_iter))

    example_data = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    ds = Dataset.Tabular.from_delimited_files(example_data, header=False)        
    x, y = clean_data(ds.to_pandas_dataframe())

    x_train, x_test,  y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=42)

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", float(accuracy))

if __name__ == '__main__':
    main()