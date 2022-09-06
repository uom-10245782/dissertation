#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 15:23:02 2022

@author: oliviadavidson
"""
# Conda
# conda install -c conda-forge xgboost
# Import necessary packages
# XGBoost 
import xgboost as xgb
# Pandas
import pandas as pd 
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
# SKLearn
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
# Plotting
from matplotlib import pyplot as plt
# Datetime
import datetime
from datetime import datetime
# Numpy
import numpy as np
# Timer 
import time

# Load data 
series = read_csv('', header=0, index_col=0)

# TimeSeriesSplit
splits = TimeSeriesSplit(n_splits=10)

# Set model - XGBoost Regression
model = xgb.XGBRegressor(objective='reg:squarederror', verbose=False)

# Exclude the last 10% of observations, which is the holdout-test set that will not be used in the CV
len(series)
# 6209
# 10% of 6209 == 621
# 20% of 6209 == 1242

# Train
train = series.iloc[:5241,:]
x_train = train.iloc[:,:48]
y_train = train.iloc[:,48:]

# Test
test = series.iloc[5241:,:]
x_test = test.iloc[:,:48]
y_test = test.iloc[:,48:]
# Evaluation set for final model
eval_set = [(x_test, y_test)]

# A parameter grid for XGBoost
pg = {
    'max_depth': [6, 7, 8, 9],
    'min_child_weight': [0.5, 1, 1.5, 2, 2.5],
    'n_estimators': [100, 200, 300, 400, 500],
    'gamma':[0.1,0.2,0.3,0.4]}


grid_search = GridSearchCV(model, param_grid=pg,  
                           n_jobs=-1, cv=splits, verbose=2)
start = time.time()
grid_result = grid_search.fit(x_train, y_train)
end = time.time()
 
print(end - start)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

results = pd.DataFrame(grid_result.cv_results_)
results.to_csv('xgb-random-grid-search-results-01.csv', index=True)

