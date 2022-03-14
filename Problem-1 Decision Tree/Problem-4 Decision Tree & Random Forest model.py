# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 11:54:24 2021

@author: chinni
"""

'''4.	In the recruitment domain, HR faces the challenge of predicting if the candidate is 
    faking their salary or not. For example, a candidate claims to have 5 years of experience
    and earns 70,000 per month working as a regional manager. The candidate expects more money
    than his previous CTC. We need a way to verify their claims (is 70,000 a month working as a
    regional manager with an experience of 5 years a genuine claim or does he/she make less than that?)
    Build a Decision Tree and Random Forest model with monthly income as the target variable'''
    
import pandas as pd
import numpy as np
profile=pd.read_csv(r"C:\data\data science\Study material\Decision Tree\Datasets_DTRF\HR_DT.csv")

dataMapping={
            'CEO':1,
            'Country Manager':2,
            'Region Manager':3,
            'Manager':4,
            'Senior Partner':5,
            'Partner':6,
            'Senior Consultant':7,
            'Junior Consultant':8,
            'C-level':9,
            'Business Analyst':10
            }
profile['Position of the employee']=profile['Position of the employee'].map(dataMapping)
profile.isna().sum()
profile.describe()
import seaborn as sns
sns.boxplot(profile['no of Years of Experience of employee'])
sns.boxplot(profile[' monthly income of employee'])
##there are no outliers and no null  values
profile.columns
# Input and Output Split
#predictors = profile['Position of the employee', 'no of Years of Experience of employee']
colnames=list(profile.columns)
predictors = colnames[:2:1]
type(predictors)
target = colnames[-1] 
type(target)
profile.info()

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train, test = train_test_split(profile, test_size = 0.33)

from sklearn.ensemble import RandomForestClassifier
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

## test accuracy is 0.81 and train accuracy is 0.96

#########applying RandomForestClassifier method#############
rf= RandomForestClassifier()
rf = RandomForestClassifier(n_jobs=2, n_estimators=15, criterion="entropy")
rf.fit(train[predictors], train[target]) # Fitting RandomForestClassifier model from sklearn.ensemble  
pred=rf.predict(test[predictors])
from sklearn.metrics import accuracy_score
print(accuracy_score(test[target],pred))
#test accuracy
test_accuracy=np.mean(rf.predict(test[predictors])==test[target])
test_accuracy
#train accuaracy
train_accuracy=np.mean(rf.predict(train[predictors])==train[target])
train_accuracy
### Train accuracy is 0.96 and test accuracy is 0.41 using random forest model.
