# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 12:47:11 2021

@author: chinni
"""


import pandas as pd
import numpy as np
forestfires= pd.read_csv(r"C:\data\data science\Study material\SVM\Datasets_SVM\forestfires.csv")
forestfires.isna().sum()
forestfires.dtypes
forestfires.drop(['month','day'], axis = 1, inplace=True)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

#Converting the nonnumeric data to the numeric data
forestfires['size_category']= le.fit_transform(forestfires['size_category'])
forestfires = forestfires.iloc[:,[28,0,1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,27]]

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

train,test = train_test_split(forestfires, test_size = 0.20)


train_X = train.iloc[:, 1:]
train_y = train.iloc[:, 0]
test_X  = test.iloc[:, 1:]
test_y  = test.iloc[:, 0]

#Generating linear kernal model

from sklearn.svm import SVC
forestfires_linear = SVC(kernel = "linear")
forestfires_linear.fit(train_X, train_y)
pred_test_linear = forestfires_linear.predict(test_X)

np.mean(pred_test_linear == test_y)

# Generating RBF kernal model
forestfires_rbf = SVC(kernel = "rbf")
forestfires_rbf.fit(train_X, train_y)
pred_test_rbf = forestfires_rbf.predict(test_X)

np.mean(pred_test_rbf==test_y)
