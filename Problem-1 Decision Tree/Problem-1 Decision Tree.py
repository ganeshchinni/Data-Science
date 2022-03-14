# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 10:18:17 2021

@author: chinni
"""

'''1.	A cloth manufacturing company is interested to know about the different attributes contributing
    to high sales. Build a decision tree & random forest model with Sales as target variable
    (first convert it into categorical variable).'''
    
    
'''1.	Business Problem
    1.1.	What is the business objective?
            objectives is to get high salse.
    1.1.	Are there any constraints?
            constraints might be  lack of advertisement's and proper display of avaliability of stocks.'''
            

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

company=pd.read_csv(r"C:\data\data science\Study material\Decision Tree\Datasets_DTRF\Company_Data.csv")

company.describe()
type(company.Sales)

# Converting the continuous variable "sales " into binary discrete:
cut_labels = [0,1]
cut_bins = [0, 8, 17]
company['Sales'] = pd.cut(company['Sales'], bins = cut_bins, labels = cut_labels)

company.isna().sum()
company.dropna()

company.Sales.unique()
company.Sales.value_counts()

#converting into binary --> Label encoding
lb = LabelEncoder()
company["Sales"] = lb.fit_transform(company["Sales"])
company["ShelveLoc"] = lb.fit_transform(company["ShelveLoc"])
company["Urban"] = lb.fit_transform(company["Urban"])
company["US"] = lb.fit_transform(company["US"])
     
colnames = list(company.columns)
predictors = colnames[1:]         # Inputs
target = colnames[0]              # Outputs

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train, test = train_test_split(company, test_size = 0.33)

from sklearn.tree import DecisionTreeClassifier as DT

model = DT(criterion = 'entropy')
model.fit(train[predictors], train[target])

# Prediction on Test Data
preds = model.predict(test[predictors])
pd.crosstab(test[target], preds, rownames=['Actual'], colnames=['Predictions'])##comparing predected values with actual target(sales) value

np.mean(preds == test[target]) # Test Data Accuracy = 0.78

# Prediction on Train Data
preds = model.predict(train[predictors])
pd.crosstab(train[target], preds, rownames = ['Actual'], colnames = ['Predictions'])

np.mean(preds == train[target]) # Train Data Accuracy

## test accuracy is 0.75 and train accuracy is 1.0
