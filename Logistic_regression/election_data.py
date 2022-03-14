# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 08:41:36 2021

@author: chinni

"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from sklearn import metrics

election_data=pd.read_csv(r"C:\data\data science\Study material\Logistic_regression\Datasets_LR\election_data.csv")
election_data.isna().sum()
election_data.drop('Election-id',axis=1,inplace=True)
election_data.rename(columns={'Amount Spent':'Amount_Spent','Popularity Rank':'Popularity_Rank'}, inplace=True)
election_data.dropna(axis=0, inplace=True)

# we have only 10 rows - so we cannot split it between train and test as it wont give ggod result
sns.countplot(election_data['Result'])

#the target column is split 40 - 60 ratio, so it is balanced
target = election_data['Result']
predictors = election_data.drop('Result', axis = 1)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaled_predictors=scaler.fit_transform(predictors)

from sklearn.linear_model import LogisticRegression
log_model=LogisticRegression()
log_model.fit(scaled_predictors, target)

##checking accuracy
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,plot_confusion_matrix
y_pred=log_model.predict(scaled_predictors)

confusion_matrix(target, y_pred)
plot_confusion_matrix(log_model, scaled_predictors, target)
print(classification_report(target, y_pred))

from sklearn.metrics import plot_roc_curve

plot_roc_curve(log_model, scaled_predictors, target)

coefs = pd.Series(index = predictors.columns, data = log_model.coef_[0])
coefs
# Coefficients of the predictor variables
# We get accuracy of 100% on the training dataset
## since this dataset is very small we cannot split it between train and test
## therfore we cannot truly test this model unless more information is available





# Model building 
# import statsmodels.formula.api as sm

logit_model = sm.logit('Result ~ Year + Amount_Spent + Popularity_Rank', data = election_data,(method='bfgs').fit()

