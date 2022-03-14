# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 11:45:17 2021

@author: chinni
"""

import pandas as pd
mdata=pd.read_csv(r"C:\data\data science\Study material\Multinomial Regression\Datasets_Multinomial\mdata.csv")
mdata.columns
mdata=mdata.drop('Unnamed: 0', axis=1)
mdata=mdata.rename(columns={'female':'gender'})

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
mdata['gender']=le.fit_transform(mdata['gender'])
mdata['ses']=le.fit_transform(mdata['ses'])
mdata['schtyp']=le.fit_transform(mdata['schtyp'])
mdata['honors']=le.fit_transform(mdata['honors'])

mdata.isna().sum()
mdata.prog.value_counts()

import seaborn as sns
#checking outliers
sns.boxplot(mdata.id)
sns.boxplot(mdata.gender)
sns.boxplot(mdata.ses)
sns.boxplot(mdata.schtyp)
sns.boxplot(mdata.prog)
sns.boxplot(mdata.read)
sns.boxplot(mdata.write)
sns.boxplot(mdata.math)
sns.boxplot(mdata.science)
# Boxplot of independent variable distribution for each category of honors

sns.boxplot(x = 'prog', y = 'id', data = mdata)
sns.boxplot(x = 'prog', y = 'gender', data = mdata)
sns.boxplot(x = 'prog', y = 'ses', data = mdata)
sns.boxplot(x = 'prog', y = 'schtyp', data = mdata)
sns.boxplot(x = 'prog', y = 'read', data = mdata)
sns.boxplot(x = 'prog', y = 'write', data = mdata)
sns.boxplot(x = 'prog', y = 'math', data = mdata)
sns.boxplot(x = 'prog', y = 'science', data = mdata)

# Scatter plot for each categorical choice of car
sns.stripplot(x = 'prog', y = 'id',jitter = True, data = mdata)
sns.stripplot(x = 'prog', y = 'gender',jitter = True, data = mdata)
sns.stripplot(x = 'prog', y = 'ses', jitter = True, data = mdata)
sns.stripplot(x = 'prog', y = 'schtyp', jitter = True, data = mdata)
sns.stripplot(x = 'prog', y = 'read', jitter = True, data = mdata)
sns.stripplot(x = 'prog', y = 'write', jitter = True, data = mdata)
sns.stripplot(x = 'prog', y = 'math', jitter = True, data = mdata)
sns.stripplot(x = 'prog', y = 'science', jitter = True, data = mdata)

# Scatter plot between each possible pair of independent variable and also histogram for each independent variable 
sns.pairplot(mdata) # Normal
sns.pairplot(mdata, hue = "prog") # With showing the category of each car choice in the scatter plot

# Correlation values between each independent features
mdata.corr()
mdata = mdata.iloc[:,[4,0,1,2,3,5,6,7,8,9]]

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
train, test = train_test_split(mdata, test_size = 0.2)

# ‘multinomial’ option is supported only by the ‘lbfgs’ and ‘newton-cg’ solvers
model = LogisticRegression(multi_class = "multinomial", solver = "newton-cg").fit(train.iloc[:, 1:], train.iloc[:, 0])
help(LogisticRegression)

test_predict = model.predict(test.iloc[:, 1:]) # Test predictions

# Test accuracy 
accuracy_score(test.iloc[:,0], test_predict)

train_predict = model.predict(train.iloc[:, 1:]) # Train predictions 
# Train accuracy 
accuracy_score(train.iloc[:,0], train_predict) 

# train and test both  acccuracy is same so that model is right fit