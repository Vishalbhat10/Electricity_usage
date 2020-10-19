#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def rmse(y_true, y_pred):
    '''Returns the root mean squared error of a regression prediction.'''
    r = np.sqrt(mse(y_true, y_pred))
    return r

def mape(y_true, y_pred):
    '''Returns the Mean Absolute Percentage Error of a regression prediction. 
    Note that for a correct result, the order of input y_true, y_pred must be respected.'''
    m = (mae(y_true, y_pred) / y_true.mean()) * 100
    return m

def print_metrics(y_true, y_pred):
    '''Prints the RMSE and MAPE of an estimator. Requires the order of input y_true, y_pred to be respected.'''
    print('The root mean squared error of the estimator is %.2f'% rmse(y_true, y_pred))
    print('The mean absolute percentage error of the estimator is %.1f percent'% mape(y_true, y_pred))

def show_acf_plots(mydata):
    '''Quick printing of autocorrelation and partial autocorrelation plots.'''
    fig = plt.figure(figsize=(16,8))
    ax1 = fig.add_subplot(211)
    fig = plot_acf(mydata, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = plot_pacf(mydata, ax=ax2)
    plt.show()
    return None

class param_tracker():
    '''An object designed to help tune XGBoost hyper-parameters in an iterative way. 
    This object keeps track of the best score for the estimator and the best hyper-parameters associated with it.
    To use this object, tune hyper-parameters one by one, each time the best previous hyper-parameters will automatically 
    be set so that a new hyper parameter can be tested.  '''
    def __init__(self, estimator, X_train, y_train, X_test, y_test):
        self.params = {}
        self.best_score_yet = 1e9999
        self.estimator = estimator
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
    def param_score_update(self):
        if self.estimator.best_score < self.best_score_yet:
            self.best_score_yet = self.estimator.best_score
            self.params = self.estimator.get_params()
            print('updated parameters: new best score = {}'.format(self.best_score_yet))           
    def test_params(self, parameter, parameter_list):
        self.estimator.set_params(**self.params)
        for aval in parameter_list:
            pardict = {}
            pardict[parameter] = aval
            self.estimator.set_params(**pardict)
            self.estimator.fit(self.X_train, self.y_train, 
                               eval_set = [(self.X_train, self.y_train), (self.X_test, self.y_test)], 
                               early_stopping_rounds = 10, verbose = False)
            print('results for {} of {}: rmse {}'.format(parameter, aval, self.estimator.best_score))
            self.param_score_update()

def plot_feature_importances(model, X):
    '''Plots feature importances of models. Assumes that the model has a feature_importances_ attribute.'''
    my_feature_importances = pd.Series(model.feature_importances_)
    my_variables = pd.Series(X.columns.values)
    feature_importance_df = pd.concat([my_variables, my_feature_importances], axis=1)
    feature_importance_df.columns = ['variables', 'importance']
    feature_importance_df = feature_importance_df.sort_values(by = 'importance', ascending=False)
    feature_importance_df.plot('variables', 'importance', 'bar')
    return None

def plot_holiday_impact(df, holiday_list):
    fig = plt.figure(figsize=(16,6))
    plt.plot(df.PJME_MW)
    for holiday in holiday_list:
        plt.axvline(holiday, color = 'r', ls = ':')
    plt.ylabel('Electricity Consumption (MW)')
    fig.show()
    return None