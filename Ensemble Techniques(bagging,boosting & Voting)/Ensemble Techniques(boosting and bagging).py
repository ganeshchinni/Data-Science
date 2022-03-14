# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 06:53:50 2021

@author: chinni
"""
#####################Ensemble Techniques####################

'''1. Given is the diabetes dataset. Build an ensemble model to correctly classify the outcome variable
        and improve your model prediction by using GridSearchCV. You must apply Bagging, Boosting,
        Stacking, and Voting on the dataset.'''
        
import pandas as pd
import numpy as np
report=pd.read_csv(r"C:\data\data science\Study material\Ensemble Techniques\Datasets_ET\Diabeted_Ensemble.csv")
a=report.columns
report=pd.get_dummies(report, columns = [' Class variable'], drop_first = True)
a=report.columns
report.isna().sum()## checking null values
###Data preprocessing EDA
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)
report.describe()# for 1 & 2
report[' Number of times pregnant'].skew()#(+ve skew)
report[' Plasma glucose concentration'].skew()#(+ve skew)
report[' Diastolic blood pressure'].skew()#(-ve skew)
report[' Triceps skin fold thickness'].skew()#(+ve skew)
report[' 2-Hour serum insulin'].skew()#(+ve skew)
report[' Body mass index'].skew()#(-ve skew)
report[' Diabetes pedigree function'].skew()#((+ve skew)
report[' Age (years)'].skew()#(+ve skew)
report[' Class variable_YES'].skew()#(+ve skew)
import seaborn as sns
sns.boxplot(report[' Number of times pregnant'])#outlier found
sns.boxplot(report[' Plasma glucose concentration'])# not found
sns.boxplot(report[' Diastolic blood pressure'])#outlier found
sns.boxplot(report[' Triceps skin fold thickness'])#not found
sns.boxplot(report[' 2-Hour serum insulin'])#outlier found
sns.boxplot(report[' Body mass index'])#outlier found
sns.boxplot(report[' Diabetes pedigree function'])#outlier found
sns.boxenplot(report[' Age (years)']) #outlier found
sns.boxenplot(report[' Class variable_YES'])# not found

#finding IQR limits

IQR1=report[' Number of times pregnant'].quantile(0.75)-report[' Number of times pregnant'].quantile(0.25)
IQR2=report[' Plasma glucose concentration'].quantile(0.75)-report[' Plasma glucose concentration'].quantile(0.25)
IQR3=report[' 2-Hour serum insulin'].quantile(0.75)-report[' 2-Hour serum insulin'].quantile(0.25)
IQR4=report[' Body mass index'].quantile(0.75)-report[' Body mass index'].quantile(0.25)
IQR5=report[' Diabetes pedigree function'].quantile(0.75)-report[' Diabetes pedigree function'].quantile(0.25)
IQR6=report[' Class variable_YES'].quantile(0.75)-report[' Class variable_YES'].quantile(0.25)

lower_limit1 = report[' Number of times pregnant'].quantile(0.25) - (IQR1 * 1.5)
upper_limit1 = report[' Number of times pregnant'].quantile(0.75) + (IQR1 * 1.5)

lower_limit2 = report[' Plasma glucose concentration'].quantile(0.25) - (IQR2 * 1.5)
upper_limit2 = report[' Plasma glucose concentration'].quantile(0.75) + (IQR2 * 1.5)

lower_limit3 = report[' 2-Hour serum insulin'].quantile(0.25) - (IQR3 * 1.5)
upper_limit3 = report[' 2-Hour serum insulin'].quantile(0.75) + (IQR3 * 1.5)

lower_limit4 = report[' Body mass index'].quantile(0.25) - (IQR4 * 1.5)
upper_limit4 = report[' Body mass index'].quantile(0.75) + (IQR4 * 1.5)

lower_limit5 = report[' Diabetes pedigree function'].quantile(0.25) - (IQR5 * 1.5)
upper_limit5 = report[' Diabetes pedigree function'].quantile(0.75) + (IQR5 * 1.5)

lower_limit6 = report[' Class variable_YES'].quantile(0.25) - (IQR6 * 1.5)
upper_limit6 = report[' Class variable_YES'].quantile(0.75) + (IQR6 * 1.5)

# Now let's replace the outliers by the maximum and minimum limit
report[' Number of times pregnant'] = pd.DataFrame(np.where(report[' Number of times pregnant'] > upper_limit1, upper_limit1, np.where(report[' Number of times pregnant'] < lower_limit1, lower_limit1, report[' Number of times pregnant'])))                             
report[' Plasma glucose concentration'] = pd.DataFrame(np.where(report[' Plasma glucose concentration'] > upper_limit2, upper_limit2, np.where(report[' Plasma glucose concentration'] < lower_limit2, lower_limit2, report[' Plasma glucose concentration'])))                             
report[' 2-Hour serum insulin'] = pd.DataFrame(np.where(report[' 2-Hour serum insulin'] > upper_limit3, upper_limit3, np.where(report[' 2-Hour serum insulin'] < lower_limit3, lower_limit3, report[' 2-Hour serum insulin'])))                             
report[' Body mass index'] = pd.DataFrame(np.where(report[' Body mass index'] > upper_limit4, upper_limit4, np.where(report[' Body mass index'] < lower_limit4, lower_limit4, report[' Body mass index'])))                             
report[' Diabetes pedigree function'] = pd.DataFrame(np.where(report[' Diabetes pedigree function'] > upper_limit5, upper_limit5, np.where(report[' Diabetes pedigree function'] < lower_limit5, lower_limit5, report[' Diabetes pedigree function'])))                             
report[' Class variable_YES'] = pd.DataFrame(np.where(report[' Class variable_YES'] > upper_limit6, upper_limit6, np.where(report[' Class variable_YES'] < lower_limit6, lower_limit6, report[' Class variable_YES'])))                             
                                                 
sns.boxplot(report[' Number of times pregnant'])
sns.boxplot(report[' Plasma glucose concentration'])
sns.boxplot(report[' 2-Hour serum insulin'])
sns.boxplot(report[' Body mass index'])
sns.boxplot(report[' Diabetes pedigree function'])
sns.boxplot(report[' Class variable_YES'])

#now outlers are removed

#input and output split
predictors = report.loc[:, report.columns!=' Class variable_YES']
type(predictors)

target = report[' Class variable_YES']
type(target)

##train and test partition of the data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(predictors,target,test_size=0.2,random_state=0)


#########3bagging classifier########

from sklearn.tree import DecisionTreeClassifier
clfTree=DecisionTreeClassifier()
from sklearn.ensemble import BaggingClassifier
bag_clsf=BaggingClassifier(base_estimator=clfTree,n_estimators=400,bootstrap=True,n_jobs=1,random_state=30)
bag_clsf.fit(x_train,y_train)
from sklearn.metrics import accuracy_score,confusion_matrix
#evaluation on testing data
confusion_matrix(y_test, bag_clsf.predict(x_test))
accuracy_score(y_test,bag_clsf.predict(x_test))

#evaluation on training data
confusion_matrix(y_train, bag_clsf.predict(x_train))
accuracy_score(y_train,bag_clsf.predict(x_train))

##now accuracy score on test data is 0.818% and on train data is 1.0%.

#############boosting classifier#############
##Grddient boosting

from sklearn.ensemble import GradientBoostingClassifier
boost_clsf=GradientBoostingClassifier()
boost_clsf.fit(x_train, y_train)
from sklearn.metrics import accuracy_score,confusion_matrix
confusion_matrix(y_test,boost_clsf.predict(x_test))
accuracy_score(y_test, boost_clsf.predict(x_test))
#hyperparameters
boost_clsf2=GradientBoostingClassifier(learning_rate=0.02,n_estimators=1000,max_depth=1)
boost_clsf2.fit(x_train, y_train)
#evaluation on test data
confusion_matrix(y_test,boost_clsf2.predict(x_test))
accuracy_score(y_test,boost_clsf2.predict(x_test))
#evaluation on train data

confusion_matrix(y_train, boost_clsf2.predict(x_train))
accuracy_score(y_train,boost_clsf2.predict(x_train))

##now accuracy score on test data is 0.805% and on train data is 0.79%
#Adaboosting technique.

from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(learning_rate = 0.02, n_estimators = 5000)

ada_clf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

# Evaluation on Testing Data
confusion_matrix(y_test, ada_clf.predict(x_test))
accuracy_score(y_test, ada_clf.predict(x_test))

# Evaluation on Training Data
confusion_matrix(y_train, ada_clf.predict(x_train))
accuracy_score(y_train, ada_clf.predict(x_train))


#XGBoosting
import xgboost as xgb

xgb_clf = xgb.XGBClassifier(max_depths = 5, n_estimators = 10000, learning_rate = 0.3, n_jobs = -1)

xgb_clf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

# Evaluation on Testing Data
confusion_matrix(y_test, xgb_clf.predict(x_test))
accuracy_score(y_test, xgb_clf.predict(x_test))

# Evaluation on Training Data
confusion_matrix(y_train, xgb_clf.predict(x_train))
accuracy_score(y_train, xgb_clf.predict(x_train))



