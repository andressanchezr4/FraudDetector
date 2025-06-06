# FraudDetector
Approach followed for the Kaggle competition IEEE-CIS Fraud Detection: [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection).

The best model (XGBoost) achieved a public score of 0.84/1.

## Script Description
* run.py --> End to end analysis.
* EDA.py --> Exploratory Data Analysis and data depuration.
* MLs.py --> Refactorised Machine Learning model implementation.
* Superlearner --> Stacked Machine Learning model implmementation.
* DLs.py --> Autoencoder implementation for anomaly detection and dimensionality reduction. 

At the end of the run.py, the best model also includes a Conformal Prediction layer to measure the uncertainty of the predictions.

## Requirements
* Numpy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* xgboost
* Tensorflow

## Disclaimer
Data must be downloaded from [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) as it was too heavy to be uploaded to this repository. 
