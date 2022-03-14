# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 14:59:25 2021

@author: chinni
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split # train and test 
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
affairs=pd.read_csv(r"C:\data\data science\Study material\Logistic_regression\Datasets_LR\Affairs.csv",sep = ",")
affairs.columns
affairs=affairs.drop('Unnamed: 0', axis=1)   #removing index column
affairs.isna().sum()
affairs.describe()
##segrating the naffairs column to binary type.
affairs["affir"]=np.where(affairs.naffairs>1,1,affairs["naffairs"])
affairs=affairs.drop('naffairs', axis=1)
##model  bulding
a=affairs.drop('affir',axis=1)
logit_model=sm.logit('affir ~ vryunhap +kids+ unhap + avgmarr + hapavg + vryhap + antirel + notrel + slghtrel + smerel + vryrel + yrsmarr1 + yrsmarr2 + yrsmarr3 + yrsmarr4 + yrsmarr5 + yrsmarr6', data = affairs).fit()

#summary of the model
logit_model.summary2()
logit_model.summary()

pred=logit_model.predict(affairs.iloc[:,0:])


# from sklearn import metrics
fpr, tpr, thresholds = roc_curve(affairs.affir, pred)
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
affairs["pred"] = np.zeros(601)
# taking threshold value and above the prob value will be treated as correct value 
affairs.loc[pred > optimal_threshold, "predictions"] = 1
# classification report
classification = classification_report(affairs["pred"], affairs["affir"])
classification


#Splitting the data inti training and testing data 
train_data, test_data = train_test_split(affairs, test_size = 0.2)

# Model building 
# import statsmodels.formula.api as sm
model = sm.logit('affir ~ kids + vryunhap + unhap + avgmarr + hapavg + vryhap + antirel + notrel + slghtrel + smerel + vryrel + yrsmarr1 + yrsmarr2 + yrsmarr3 + yrsmarr4 + yrsmarr5 + yrsmarr6', data = train_data).fit()

#summerising the model 
model.summary2()
model.summary()

# Prediction on Test data set
test_pred = logit_model.predict(test_data)

# Creating new column for storing predicted class of Attorney
# filling all the cells with zeroes
test_data["test_pred"] = np.zeros(121)

# taking threshold value as 'optimal_threshold' and above the thresold prob value will be treated as 1 
test_data.loc[test_pred > optimal_threshold, "test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['affir'])
confusion_matrix

accuracy_test = (94 + 4)/(121) 
accuracy_test

# classification report
classification_test = classification_report(test_data["test_pred"], test_data["affir"])
classification_test

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(test_data["affir"], test_pred)

#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")

roc_auc_test = metrics.auc(fpr, tpr)
roc_auc_test


# prediction on train data
train_pred = model.predict(train_data.iloc[ :, 0: ])

# Creating new column 
# filling all the cells with zeroes
train_data["train_pred"] = np.zeros(480)

# taking threshold value and above the prob value will be treated as correct value 
train_data.loc[train_pred > optimal_threshold, "train_pred"] = 1

# confusion matrix
confusion_matrx = pd.crosstab(train_data.train_pred, train_data['affir'])
confusion_matrx

accuracy_train = (388 + 20)/(480)
print(accuracy_train)

