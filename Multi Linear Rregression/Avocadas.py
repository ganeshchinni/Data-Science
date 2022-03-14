# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 22:13:43 2021

@author: chinni
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
# creating instance for one hot encoding
from sklearn.preprocessing import LabelEncoder

# loading the data set into python
avacado = pd.read_csv(r"C:\data\data science\Study material\Multi Linear aregression\Datasets_MLR\Avacado_Price.csv")
le = LabelEncoder()

avacado['type']=le.fit_transform(avacado['type'])
avacado['region']=le.fit_transform(avacado['region'])
a=avacado.columns
avacado.describe()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

# AveragePrice
plt.bar(height = avacado.AveragePrice, x = np.arange(1, 18250, 1))
plt.hist(avacado.AveragePrice) #histogram
plt.boxplot(avacado.AveragePrice) #boxplot

# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(avacado.AveragePrice, dist = "norm", plot = pylab)
plt.show()

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(avacado.iloc[:, :])
                             
# Correlation matrix 
b=avacado.corr()

# we see there exists High collinearity between input variables especially between
# [total volume & small bags & total volume],so there exists collinearity problem

# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
ml1 = smf.ols('AveragePrice ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Total_Bags + Small_Bags + Large_Bags', data = avacado).fit() # regression model

# Summary
ml1.summary()
# p-values are less than 0.05,

# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1)
# Studentized Residuals = Residual/standard deviation of residuals

# Preparing model                  
ml_new = smf.ols('AveragePrice ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Total_Bags + Small_Bags + Large_Bags', data = avacado).fit()    

# Summary
ml_new.summary()

# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables
rsq_Price = smf.ols('AveragePrice ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Total_Bags + Small_Bags + Large_Bags', data = avacado).fit().rsquared  
vif_Price = 1/(1 - rsq_Price) 

rsq_Volume = smf.ols('Total_Volume ~ AveragePrice + tot_ava1 + tot_ava2 + tot_ava3 + Total_Bags + Small_Bags + Large_Bags', data = avacado).fit().rsquared  
vif_Volume = 1/(1 - rsq_Volume)

rsq_ava1 = smf.ols('tot_ava1 ~ AveragePrice + Total_Volume + tot_ava2 + tot_ava3 + Total_Bags + Small_Bags + Large_Bags', data = avacado).fit().rsquared  
vif_ava1 = 1/(1 - rsq_ava1) 

rsq_ava2 = smf.ols('tot_ava2 ~ AveragePrice + Total_Volume + tot_ava1 + tot_ava3 + Total_Bags + Small_Bags + Large_Bags', data = avacado).fit().rsquared  
vif_ava2 = 1/(1 - rsq_ava2) 

rsq_ava3 = smf.ols('tot_ava3 ~ AveragePrice + Total_Volume + tot_ava2 + tot_ava1 + Total_Bags + Small_Bags + Large_Bags', data = avacado).fit().rsquared  
vif_ava3 = 1/(1 - rsq_ava3) 

rsq_total = smf.ols('Total_Bags ~ AveragePrice + Total_Volume + tot_ava2 + tot_ava1 + tot_ava3 + Small_Bags + Large_Bags', data = avacado).fit().rsquared  
vif_total = 1/(1 - rsq_total) 

rsq_small = smf.ols('Small_Bags ~ AveragePrice + Total_Volume + tot_ava2 + tot_ava1 + tot_ava3 + Total_Bags + Large_Bags', data = avacado).fit().rsquared  
vif_small = 1/(1 - rsq_small) 

rsq_large = smf.ols('Large_Bags ~ AveragePrice + Total_Volume + tot_ava2 + tot_ava1 + tot_ava3 + Small_Bags + Total_Bags', data = avacado).fit().rsquared  
vif_large = 1/(1 - rsq_large) 


# Storing vif values in a data frame
d1 = {'Variables':['AveragePrice' , 'Total_Volume' , 'tot_ava1', 'tot_ava2', 'tot_ava3', 'Total_Bags', 'Small_Bags' ,'Large_Bags'], 'VIF':[vif_Price, vif_Volume, vif_ava1, vif_ava2, vif_ava3,vif_total, vif_small,vif_large]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# As WT is having highest VIF value, we are going to drop this from the prediction model

# Final model
final_ml = smf.ols('AveragePrice ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Total_Bags + Small_Bags + Large_Bags', data = avacado).fit()
final_ml.summary() 

# Prediction
pred = final_ml.predict(avacado)

# Q-Q plot
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

import seaborn as sns
# Residuals vs Fitted plot
sns.residplot(x = pred, y = avacado.Price, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(final_ml)


### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
avacado_train, avacado_test = train_test_split(avacado, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols('AveragePrice ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Total_Bags + Small_Bags + Large_Bags', data = avacado_train).fit()

# prediction on test data set 
test_pred = model_train.predict(avacado_test)

# test residual values 
test_resid = test_pred - avacado_test.AveragePrice
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse


# train_data prediction
train_pred = model_train.predict(avacado_train)

# train residual values 
train_resid  = train_pred - avacado_train.AveragePrice
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse
