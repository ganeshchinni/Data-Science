# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 17:37:13 2021

@author: chinni
"""

import pandas as pd
startup=pd.read_csv(r"C:\data\data science\Study material\Multi Linear aregression\Datasets_MLR\50_Startups.csv")
startup=pd.get_dummies(startup, drop_first=True)
startup.columns
startup.isna().sum()
startup.duplicated().sum()
#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 
import numpy as np

# this will modify the name of the first column
startup.rename(columns={'R&D_Spend':'R_D','State_New York':'State_New_York'},inplace=True)
a=startup.columns
startup.R_D
#checking for 'R&D Spend'
plt.bar(height = startup["R_D"], x = np.arange(1, 51, 1))
plt.hist(startup["R_D"]) #histogram
plt.boxplot(startup["R_D"]) #boxplot

#checking for 'R&D Spend'
plt.bar(height = startup["Administration"], x = np.arange(1, 51, 1))
plt.hist(startup["Administration"]) #histogram
plt.boxplot(startup["Administration"]) #boxplot

#checking for 'Marketing Spend'
plt.bar(height = startup["Marketing_Spend"], x = np.arange(1, 51, 1))
plt.hist(startup["Marketing_Spend"]) #histogram
plt.boxplot(startup["Marketing_Spend"]) #boxplot

#checking for 'Profit'
plt.bar(height = startup["Profit"], x = np.arange(1, 51, 1))
plt.hist(startup["Profit"]) #histogram
plt.boxplot(startup["Profit"]) #boxplot

#checking for 'State_Florida'
plt.bar(height = startup["State_Florida"], x = np.arange(1, 51, 1))
plt.hist(startup["State_Florida"]) #histogram
plt.boxplot(startup["State_Florida"]) #boxplot

#checking for 'State_New York'
plt.bar(height = startup["State_New_York"], x = np.arange(1, 51, 1))
plt.hist(startup["State_New_York"]) #histogram
plt.boxplot(startup["State_New_York"]) #boxplot


# Jointplot
import seaborn as sns
sns.jointplot(x=startup['R_D'], y=startup['Profit'])

# countplot for r&d
plt.figure(1, figsize=(6,10))
sns.countplot(startup['R_D'])

#countplot for profit
plt.figure(1, figsize=(6,10))
sns.countplot(startup['Profit'])

# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(startup.R_D, dist = "norm", plot = pylab)
stats.probplot(startup.Profit, dist = "norm", plot = pylab)

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(startup.iloc[:, :])

# Correlation matrix 
startup.corr()
# we see there exists High collinearity between input variables especially between
# [R_D & profit]

# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
ml1 = smf.ols('Profit ~ R_D + Administration + Marketing_Spend + State_Florida + State_New_York', data = startup).fit() # regression model

# Summary
ml1.summary()
# p-values for Administration,Marketing_Spend, State_Florida,State_New_York are more than 0.05

# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm
sm.graphics.influence_plot(ml1)
# Studentized Residuals = Residual/standard deviation of residuals
# index 49 is showing high influence so we can exclude that entire row

startup_new = startup.drop(startup.index[[49]])

# Preparing model                  
ml1_new = smf.ols('Profit ~ R_D + Administration + Marketing_Spend + State_Florida +State_New_York', data = startup_new).fit() # regression model
ml1_new.summary()

# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables
rsq_R_D = smf.ols('R_D ~ Administration + Marketing_Spend +Profit + State_Florida + State_New_York', data = startup).fit().rsquared  
vif_R_D = 1/(1 - rsq_R_D) 
rsq_Administration = smf.ols('Administration ~ R_D + Marketing_Spend + Profit + State_Florida', data = startup).fit().rsquared  
vif_Administration = 1/(1 - rsq_Administration) 
rsq_Marketing_Spend = smf.ols('Marketing_Spend ~ R_D + Marketing_Spend + Profit + State_Florida', data = startup).fit().rsquared  
vif_Marketing_Spend = 1/(1 - rsq_Marketing_Spend) 
rsq_State_Profit = smf.ols('Profit ~ R_D + Administration + Marketing_Spend + State_Florida', data = startup).fit().rsquared  
vif_State_Profit = 1/(1 - rsq_State_Profit) 


# Storing vif values in a data frame
d1 = {'Variables':['Profit', 'R_D', 'Administration', 'Marketing_Spend'], 'VIF':[vif_State_Profit, vif_R_D, vif_Administration, vif_Marketing_Spend]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# As Profit is having highest VIF value, we are going to drop this from the prediction model
# Final model
final_ml = smf.ols('R_D ~ Administration + Marketing_Spend + State_Florida + State_New_York', data = startup).fit()
final_ml.summary() 

# Prediction
pred = final_ml.predict(startup)

# Q-Q plot
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = startup.Profit, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(final_ml)


### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
startup_train, startup_test = train_test_split(startup, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols("Profit ~ R_D + Administration + Marketing_Spend", data = startup_train).fit()

# prediction on test data set 
test_pred = model_train.predict(startup_test)

# test residual values 
test_resid = test_pred - startup_test.Profit
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse


# train_data prediction
train_pred = model_train.predict(startup_train)

# train residual values 
train_resid  = train_pred - startup_train.Profit
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse
