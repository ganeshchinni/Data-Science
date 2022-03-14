# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 12:33:51 2021

@author: chinni
"""

import pandas as pd
import numpy as np
salary_train= pd.read_csv(r"C:\data\data science\Study material\SVM\Datasets_SVM\SalaryData_Train (1).csv")
salary_test=pd.read_csv(r"C:\data\data science\Study material\SVM\Datasets_SVM\SalaryData_Test (1).csv")

salary_train.isna().sum()
salary_test.isna().sum()
salary_train.dtypes

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

#Converting the nonnumeric data to the numeric data
salary_train['workclass']= le.fit_transform(salary_train['workclass'])
salary_train['education']= le.fit_transform(salary_train['education'])
salary_train['maritalstatus']= le.fit_transform(salary_train['maritalstatus'])
salary_train['occupation']= le.fit_transform(salary_train['occupation'])
salary_train['relationship']= le.fit_transform(salary_train['relationship'])
salary_train['race']= le.fit_transform(salary_train['race'])
salary_train['sex']= le.fit_transform(salary_train['sex'])
salary_train['native']= le.fit_transform(salary_train['native'])
salary_train['Salary']= le.fit_transform(salary_train['Salary'])

salary_test['workclass']= le.fit_transform(salary_test['workclass'])
salary_test['education']= le.fit_transform(salary_test['education'])
salary_test['maritalstatus']= le.fit_transform(salary_test['maritalstatus'])
salary_test['occupation']= le.fit_transform(salary_test['occupation'])
salary_test['relationship']= le.fit_transform(salary_test['relationship'])
salary_test['race']= le.fit_transform(salary_test['race'])
salary_test['sex']= le.fit_transform(salary_test['sex'])
salary_test['native']= le.fit_transform(salary_test['native'])
salary_test['Salary']= le.fit_transform(salary_test['Salary'])

salary_train = salary_train.iloc[:,[13,0,1,2,3,5,6,7,8,9,10,11,12]]
salary_test = salary_test.iloc[:,[13,0,1,2,3,5,6,7,8,9,10,11,12]]

'''
train_X = salary_train.iloc[:, 0:]
train_y = salary_train.iloc[:, -1]
test_X  = salary_test.iloc[:, -1:]
test_y  = salary_test.iloc[:, -1]
'''

train_X = salary_train.iloc[:, 1:] # from 1 to all columns
train_y = salary_train.iloc[:, 0] # only 0 column
test_X  = salary_test.iloc[:, 1:]
test_y  = salary_test.iloc[:, 0]

#Generating linear kernal model

from sklearn.svm import SVC
salary_linear = SVC(kernel = "linear")
salary_linear.fit(train_X, train_y)
pred_test_linear = salary_linear.predict(test_X)

np.mean(pred_test_linear == test_y)

# Generating RBF kernal model
salary_rbf = SVC(kernel = "rbf")
salary_rbf.fit(train_X, train_y)
pred_test_rbf = salary_rbf.predict(test_X)

np.mean(pred_test_rbf==test_y)

