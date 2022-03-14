# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 13:21:29 2021

@author: chinni
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets, linear_model, svm, neighbors, naive_bayes
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score


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

predictors = report.loc[:, report.columns!=' Class variable_YES']
type(predictors)

target = report[' Class variable_YES']
type(target)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.2, random_state=0)
#instantiate the learners(classifiers)
learner1=neighbors.KNeighborsClassifier(n_neighbors=5)
learner2=linear_model.Perceptron(tol=1e-2,random_state=0)
learner3=svm.SVC(gamma=0.001)
#instantiate the voting classifier
voting=VotingClassifier([('KNN',learner1),
                         ('Prc',learner2),
                         ('SVM',learner3)])
#fit classifier with voting classifier
voting.fit(x_train,y_train)
#predict the most voted classes
hard_predictions=voting.predict(x_test)
#accuracy of hard voting
print('Hard Voting:', accuracy_score(y_test, hard_predictions))


##soft voting
#instantiate the learners (classifiers)
learner4=neighbors.KNeighborsClassifier(n_neighbors=5)
learner5=naive_bayes.GaussianNB()
learner6=svm.SVC(gamma=0.001, probability=True)

# Instantiate the voting classifier
voting = VotingClassifier([('KNN', learner_4),
                           ('NB', learner_5),
                           ('SVM', learner_6)],
                            voting = 'soft')

# Fit classifier with the training data
voting.fit(x_train, y_train)
learner4.fit(x_train, y_train)
learner5.fit(x_train, y_train)
learner6.fit(x_train, y_train)

# Predict the most probable class
soft_predictions = voting.predict(x_test)

# Get the base learner predictions
predictions4 = learner4.predict(x_test)
predictions5 = learner5.predict(x_test)
predictions6 = learner6.predict(x_test)

# Accuracies of base learners
print('L4:', accuracy_score(y_test, predictions4))
print('L5:', accuracy_score(y_test, predictions5))
print('L6:', accuracy_score(y_test, predictions6))

# Accuracy of Soft voting
print('Soft Voting:', accuracy_score(y_test, soft_predictions))
## so soft voting for the given data set=0.77%