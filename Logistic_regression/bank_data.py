# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 09:43:15 2021

@author: chinni
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report

#Loading the data set
bank_data1 = pd.read_csv(r"C:\data\data science\Study material\Logistic_regression\Datasets_LR\bank_data.csv", sep = ",")
bank_data1.rename(columns={'joadmin.':'joadmin','joblue.collar':'joblue_collar','joself.employed':'joself_employed'}, inplace=True)

bank_data1.describe()

#Identifying the missing values
bank_data1.isna().sum()

#Building the basic logistic model
logistic_model = sm.logit('y ~ age + default + balance + housing + loan + duration + campaign + pdays + previous + poutfailure + poutother + poutsuccess + poutunknown + con_cellular + con_telephone + con_unknown + divorced + married + single + joadmin + joblue_collar + joentrepreneur + johousemaid + jomanagement + joretired + joself_employed + joservices + jostudent + jotechnician + jounemployed + jounknown', data = bank_data1).fit()

#summary of the model
logistic_model.summary2()
logistic_model.summary()

#Prediction of the model

bank_data1 = bank_data1.iloc[:,[31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]]
prediction = logistic_model.predict(bank_data1.iloc[ :, 0: ])

#target = bank_data1['y']
#predictors = bank_data1.drop('y', axis = 1)

#Finding fpr, tpr and threshold values
fpr, tpr, thresholds = roc_curve(bank_data1.y, prediction)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
optimal_threshold

import pylab as pl

i = np.arange(len(tpr))
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]

# Plotting tpr vs 1-fpr
fig, ax = pl.subplots()
pl.plot(roc['tpr'], color = 'red')
pl.plot(roc['1-fpr'], color = 'blue')
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
ax.set_xticklabels([])

roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)

# creating the prediction column and filling with zeros
bank_data1["prediction"] = np.zeros(45211)
# taking threshold value and above the prob value will be treated as correct value 
bank_data1.loc[prediction > optimal_threshold, "prediction"] = 1
# classification report
classification = classification_report(bank_data1["prediction"], bank_data1["y"])
classification


#Splitting the data inti training and testing data 
train_data, test_data = train_test_split(bank_data1, test_size = 0.2)

# Model building 
# import statsmodels.formula.api as sm
model = sm.logit('y ~ age + default + balance + housing + loan + duration + campaign + pdays + previous + poutfailure + poutother + poutsuccess + poutunknown + con_cellular + con_telephone + con_unknown + divorced + married + single + joadmin + joblue_collar + joentrepreneur + johousemaid + jomanagement + joretired + joself_employed + joservices + jostudent + jotechnician + jounemployed + jounknown', data = bank_data1).fit()

#summerising the model 
model.summary2()
model.summary()

# Prediction on Test data set
test_pred = logistic_model.predict(test_data)

# Creating new column for storing predicted class of Attorney
# filling all the cells with zeroes
test_data["test_pred"] = np.zeros(9043)

# taking threshold value as 'optimal_threshold' and above the thresold prob value will be treated as 1 
test_data.loc[test_pred > optimal_threshold, "test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['y'])
confusion_matrix

accuracy_test = (6507 + 881)/(9043) 
accuracy_test

# classification report
classification_test = classification_report(test_data["test_pred"], test_data["y"])
classification_test

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(test_data["y"], test_pred)

#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")

roc_auc_test = metrics.auc(fpr, tpr)
roc_auc_test


# prediction on train data
train_pred = model.predict(train_data.iloc[ :, 0: ])

# Creating new column 
# filling all the cells with zeroes
train_data["train_pred"] = np.zeros(36168)

# taking threshold value and above the prob value will be treated as correct value 
train_data.loc[train_pred > optimal_threshold, "train_pred"] = 1

# confusion matrix
confusion_matrx = pd.crosstab(train_data.train_pred, train_data['y'])
confusion_matrx

accuracy_train = (26298 + 3422)/(36168)
print(accuracy_train)

