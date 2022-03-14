# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 10:31:25 2021

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

Add=pd.read_csv(r"C:\data\data science\Study material\Logistic_regression\Datasets_LR\advertising.csv")
Add.isna().sum()
Add.columns
Add.info()
Add.drop(['Ad_Topic_Line','City','Country','Timestamp'],axis=1, inplace=True)
Add.rename(columns={'Daily_Time_ Spent _on_Site':'Daily_Time_Spent_on_Site','Daily Internet Usage':'Daily_Internet_Usage'},inplace=True)
logit_model = sm.logit('Clicked_on_Ad ~ Daily_Time_Spent_on_Site + Age + Area_Income + Daily_Internet_Usage + Male', data = Add).fit()

#summary
logit_model.summary2() # for AIC
logit_model.summary()

pred = logit_model.predict(Add.iloc[ :, 0: ])

#Finding fpr, tpr and threshold values
fpr, tpr, thresholds = roc_curve(Add.Clicked_on_Ad, pred)
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
Add["pred"] = np.zeros(1000)
# taking threshold value and above the prob value will be treated as correct value 
Add.loc[pred > optimal_threshold, "pred"] = 1
# classification report
classification = classification_report(Add["pred"], Add["Clicked_on_Ad"])
classification


#Splitting the data inti training and testing data 
train_data, test_data = train_test_split(Add, test_size = 0.2)

# Model building 
# import statsmodels.formula.api as sm
model = sm.logit('Clicked_on_Ad ~ Daily_Time_Spent_on_Site + Age + Area_Income + Daily_Internet_Usage + Male', data = Add).fit()

#summerising the model 
model.summary2()
model.summary()

# Prediction on Test data set
test_pred = logit_model.predict(test_data)

# Creating new column for storing predicted class of Attorney
# filling all the cells with zeroes
test_data["test_pred"] = np.zeros(200)

# taking threshold value as 'optimal_threshold' and above the thresold prob value will be treated as 1 
test_data.loc[test_pred > optimal_threshold, "test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['Clicked_on_Ad'])
confusion_matrix

accuracy_test = (75 + 28)/(200) 
accuracy_test

# classification report
classification_test = classification_report(test_data["test_pred"], test_data["Clicked_on_Ad"])
classification_test

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(test_data["Clicked_on_Ad"], test_pred)

#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")

roc_auc_test = metrics.auc(fpr, tpr)
roc_auc_test


# prediction on train data
train_pred = model.predict(train_data.iloc[ :, 0: ])

# Creating new column 
# filling all the cells with zeroes
train_data["train_pred"] = np.zeros(800)

# taking threshold value and above the prob value will be treated as correct value 
train_data.loc[train_pred > optimal_threshold, "train_pred"] = 1

# confusion matrix
confusion_matrx = pd.crosstab(train_data.train_pred, train_data['Clicked_on_Ad'])
confusion_matrx

accuracy_train = (448 + 256)/(800)
print(accuracy_train)
