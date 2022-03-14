# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 12:05:01 2021

@author: chinni
"""


'''3.	Build a Decision Tree & Random Forest model on the fraud data.
    Treat those who have taxable_income <= 30000 as Risky and others as Good (discretize the taxable '''


from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
data = pd.read_csv(r"C:\data\data science\Study material\Decision Tree\Datasets_DTRF\Fraud_check.csv")
data.info()
data.isna().sum()
data.dtypes
lb = LabelEncoder()
data['Undergrad'] = lb.fit_transform(data['Undergrad'])
data['Marital.Status'] = lb.fit_transform(data['Marital.Status'])
data['Urban'] = lb.fit_transform(data['Urban'])

# discritization
data.describe()
import numpy as np
data['TaxableIncome'] = np.where(data['TaxableIncome'] <= 30000, 'Risky', 'Good')
dataNew=data.iloc[:,[2,0,1,3,4,5]]
colnames=list(dataNew.columns)
predictors=colnames[1: :1]
target=colnames[0]
#splitting newData into test and train 
from sklearn.model_selection import train_test_split
train,test=train_test_split(dataNew,test_size=0.2)
########using decision tree model#####
from sklearn.tree import DecisionTreeClassifier as dt
newData=dt(criterion="entropy")
newData.fit(train[predictors],train[target])
##predection on test data
preds=newData.predict(test[predictors])
pd.crosstab(test[target],preds,rownames=['actual'],colnames=['predictions'])
np.mean(preds==test[target])
##predection train data
preds=newData.predict(train[predictors])
pd.crosstab(train[target], preds,rownames=["actual"],colnames=["predictions"])
np.mean(preds==train[target])

### Train accuracy is 1 and test accuracy is 0.68 using random forest model.


#######using RandomForestClassier##########
from sklearn.ensemble import RandomForestClassifier

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

### Train accuracy is 0.98.3 and test accuracy is 0.70 using random forest model.

