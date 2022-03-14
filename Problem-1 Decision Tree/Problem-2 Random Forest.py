# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 08:38:24 2021

@author: chinni
"""

'''2.Divide the diabetes data into train and test datasets and build a Random Forest
    and Decision Tree model with Outcome (Class ariable) as the output variable. '''
    
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
report=pd.read_csv(r"C:\data\data science\Study material\Decision Tree\Datasets_DTRF\Diabetes.csv")
#converting non numeric data into numeric by labelencoding
lb=LabelEncoder()
report[' Class variable']=lb.fit_transform(report[' Class variable'])
report.isna().sum()
#input and output split
colnames = list(report.columns)
predictors = colnames[:8] # Inputs
target = colnames[-1] # Outputs

#train and test partition of the data
from sklearn.model_selection import train_test_split
train,test=train_test_split(report,test_size=0.2)
from sklearn.ensemble import RandomForestClassifier

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
### Train accuracy is 0.99 and test accuracy is 0.75 using random forest model.


#########applying decision tree model###########
from sklearn.tree import DecisionTreeClassifier as dt
report1=dt(criterion='entropy')
report1.fit(train[predictors], train[target])

##predection on test data
preds=report1.predict(test[predictors])
pd.crosstab(test[target],preds,rownames=['Actual'],colnames=['predictions'])
np.np.mean(preds==test[target])

##prediction on train data
preds=report1.predict(train[predictors])
pd.crosstab(train[target],preds,rownames=['actual'],colnames=['redictions'])
np.np.mean(preds==train[target])

### Train accuracy is 0.70 and test accuracy is 1.0 using DecisionTreeClassifier.
