# Azure Machine Learning Engineer Capstone Project

## Table of contents
   * [Project Overview](#Project-Overview)
   * [Project requirements](#Project-requirements)
   * [Dataset](#Dataset)
   * [AutoML](#AutoML)
   * [AutoML Results](#AutoML-Results)
   * [Deployment](#Deployment)
   * [HyperDrive](#HyperDrive)
   * [HyperDrive results](#HyperDrive-results)
   * [Screen Recording](#Screen-Recording)
   * [Standout Suggestions](#Standout-Suggestions)

## Project Overview
Capstone project for Udacity Azure Machine Learning Engineer NanoDegree. 

The problem is to create programmatically an AutoML experiement and a HyperDrive Experiment from Jupyter notebooks and deploy one of the models to create an endpoint and consume that endpoint with POST request. 

***
## Project requirements
* An Azure ML workspace
* A compute instance
* Jupyter lab.

***
## Dataset

Dataset for this project is Iris for classifying Iris plant. 

https://archive.ics.uci.edu/ml/datasets/iris

Data has four columns describing the characteristics of the plant and one more column for the plant class. There are three different classes thereby making this a multiclass classification. 

***
## AutoML

To train an automl model a few steps need to be executed to setup the experiement. 

1. We setup the workspace.
2. Create an experiment
3. Setup the compute
4. Import the dataset into the workspace and register to datasets using Dataset package
5. Create automl settings with parameters to timeout, concurrent iterations and metric to use
6. Create automl config to use compute, data, target and early stopping
7. Submit the experiment and wait for completion. 

*** 
## AutoML Results

Automl model achieved a score of 0.9972 using a stack ensemble.
![Diagram](./screenshots/7.PNG?raw=true "Results of automl")
![Diagram](./screenshots/10.PNG?raw=true "Results of automl")
![Diagram](./screenshots/8.PNG?raw=true "Results of automl experiment")

*** 
## Deployment

Input:

```
{
  "data":
    [
        { 
            "Column1":2.3,
            "Column2":3.4,
            "Column3":2.9,
            "Column4":5.6
        }
    ]
}
```

The automl model was deployed to create an endpoint to utilise as rest api and sent post requests to the endpoint. 
![Diagram](./screenshots/9.PNG?raw=true "Endpoint")

***
## HyperDrive:

To create a hyperdrove experiment a few steps need to be executed to setup the experiement. 

1. We setup the workspace.
2. Create an experiment
3. Setup the compute
4. Import the dataset into the workspace and register to datasets using Dataset package
5. Define an early termination policy using the BanditPolicy
6. Use RandomParameterSampling with different parameters using --C and --max_iter of logistic regression. 
```
"--C": uniform(0.1, 1),
"--max_iter": choice(50, 75, 100, 150)
````
7. Use the training script to create an estimator. 
8. Create HyperDrive Config with the earlier parameters like estimators, policy, hyperparameter sampling, primary metric goal. 
9. Submit the experiment and wait for completion
10. Get the best model and save. 

![Diagram](./screenshots/12.PNG?raw=true "Training result")
![Diagram](./screenshots/13.PNG?raw=true "Training result")
![Diagram](./screenshots/14.PNG?raw=true "Training result")
***
## HyperDrive results

Hyperdrive results gave an accuracy of 0.9777 as indicated below. 
```
[{'run_id': 'HD_78adb20a-9fa6-4160-8e0b-5fefb096f3b5_14', 'hyperparameters': '{"--C": 0.972741918863652, "--max_iter": 150}', 'best_primary_metric': 0.9777777777777777, 'status': 'Completed'}]
```
![Diagram](./screenshots/15.PNG?raw=true "Training result")


## Screen Recording

Look at the screenshots folder for more images. This screen recording gives a short overview of the project in action.

https://youtu.be/9EVrL1yUTq4

## Standout Suggestions

1. Creating newer features with existing features. 
2. Training hyperdrive experiment for more parameters.
3. Automl model can run for longer to get better models and more options with higher accuracy. 
4. Having better computing power for decreasing latency. 