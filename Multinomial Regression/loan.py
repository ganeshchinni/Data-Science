# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 13:44:15 2021

@author: chinni
"""

import pandas as pd
loan = pd.read_csv("C:\data\data science\Study material\Multinomial Regression\Datasets_Multinomial\loan.csv")
loan.columns
loan.isna().sum()
# data preprocessing
loan1=loan.dropna(axis=1)
loan1.isna().sum()
loan1.info()
#labelenocoding for non numeric data columns
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
loan1["term"] = LE.fit_transform(loan1["term"])
loan1["int_rate"] = LE.fit_transform(loan1["int_rate"])
loan1["grade"] = LE.fit_transform(loan1["grade"])
loan1["sub_grade"] = LE.fit_transform(loan1["sub_grade"])
loan1["annual_inc"] = LE.fit_transform(loan1["annual_inc"])
loan1["home_ownership"] = LE.fit_transform(loan1["home_ownership"])
loan1["verification_status"] = LE.fit_transform(loan1["verification_status"])
loan1["home_ownership"] = LE.fit_transform(loan1["home_ownership"])
loan1["loan_status"] = LE.fit_transform(loan1["loan_status"])
loan1["pymnt_plan"] = LE.fit_transform(loan1["pymnt_plan"])
loan1["url"] = LE.fit_transform(loan1["url"])
loan1["purpose"] = LE.fit_transform(loan1["purpose"])
loan1["addr_state"] = LE.fit_transform(loan1["addr_state"])
loan1["earliest_cr_line"] = LE.fit_transform(loan1["earliest_cr_line"])
loan1["earliest_cr_line"] = LE.fit_transform(loan1["earliest_cr_line"])
loan1["application_type"] = LE.fit_transform(loan1["application_type"])
loan1["initial_list_status"] = LE.fit_transform(loan1["initial_list_status"])
loan1["issue_d"] = LE.fit_transform(loan1["issue_d"])
loan1["zip_code"] = LE.fit_transform(loan1["zip_code"])

# Boxplot for all Column's
import matplotlib.pyplot as plt
import seaborn as sns
loan1.columns
plt.figure(figsize = ( 25,5))


'''
for i,col in zip(range(1,38),loan1.columns):
    plt.subplot(1,38,i)
    sns.boxplot(x = loan1.iloc[:,[i]],data = loan1)
    plt.title(f"box plot of {col}") #no outliers


boxplot = loan1.boxplot(column=['id', 'member_id', 'loan_amnt', 'funded_amnt', 'funded_amnt_inv',
       'term', 'int_rate', 'installment', 'grade', 'sub_grade',
       'home_ownership', 'annual_inc', 'verification_status', 'issue_d',
       'loan_status', 'pymnt_plan', 'url', 'purpose', 'zip_code', 'addr_state',
       'dti', 'delinq_2yrs', 'earliest_cr_line', 'inq_last_6mths', 'open_acc',
       'pub_rec', 'revol_bal', 'total_acc', 'initial_list_status', 'out_prncp',
       'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp',
       'total_rec_int', 'total_rec_late_fee', 'recoveries',
       'collection_recovery_fee', 'last_pymnt_amnt', 'policy_code',
       'application_type', 'acc_now_delinq', 'delinq_amnt'])
'''


# Boxplot of independent variable distribution for each category of choice 
sns.boxplot(x = "loan_status", y = "funded_amnt", data = loan1)
sns.boxplot(x = "loan_status", y = "revol_bal", data = loan1)
sns.boxplot(x = "loan_status", y = "total_pymnt", data = loan1)
sns.boxplot(x = "loan_status", y = "total_pymnt_inv", data = loan1)
sns.boxplot(x = "loan_status", y = "total_rec_prncp", data = loan1)
sns.boxplot(x = "loan_status", y = "total_rec_int", data = loan1)
sns.boxplot(x = "loan_status", y = "recoveries", data = loan1)

# Scatter plot for each categorical choice of car
sns.stripplot(x = "loan_status", y = "funded_amnt", jitter = True, data = loan1)
sns.stripplot(x = "loan_status", y = "revol_bal", jitter = True, data = loan1)
sns.stripplot(x = "loan_status", y = "total_pymnt", jitter = True, data = loan1)
sns.stripplot(x = "loan_status", y = "total_pymnt_inv", jitter = True, data = loan1)
sns.stripplot(x = "loan_status", y = "total_rec_prncp", jitter = True, data = loan1)
sns.stripplot(x = "loan_status", y = "total_rec_int", jitter = True, data = loan1)
sns.stripplot(x = "loan_status", y = "recoveries", jitter = True, data = loan1)


#scatter plot between each possible pair of independent variables and histogram
sns.set(font_scale=1.5)
sns.pairplot(loan1)

#assigning 'X' inputs and 'y' outputs from the data set
x=loan1.drop(columns=['loan_status'],axis=1)
y=loan1['loan_status']

# Splitting the data into Train and test
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split( x, y, test_size = 0.30, random_state=42)

#splitting the data into test and train
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
clf=LogisticRegressionCV(multi_class='multinomial', solver= 'newton-cg')
model=clf.fit(x_train,y_train)
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#test data predection
y_pred=model.predict(x_test)

acc_test_pred=accuracy_score(y_test,y_pred)
acc_test_pred
# getting 99% Accuracy on test data
conf_mat_test = confusion_matrix(y_test,y_pred)
conf_mat_test

# Training data Prediction
x_pred = model.predict(x_train)

accu_train_pred = accuracy_score(y_train,x_pred)
accu_train_pred 

# Getting 100 % on Train Data

conf_mat_train = confusion_matrix(y_train,x_pred)
conf_mat_train
