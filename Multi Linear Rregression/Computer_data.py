# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 15:28:45 2021

@author: chinni
"""

import pandas as pd
import numpy as np
computer_data=pd.read_csv(r"C:\data\data science\Study material\Multi Linear aregression\Datasets_MLR\Computer_Data.csv")
computer_data.rename(columns={'Unnamed: 0':'x'},inplace=True)
computer_data.isna().sum()
computer_data.duplicated().sum()
computer_data.drop('x', axis=1, inplace=True)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
computer_data['cd']=le.fit_transform(computer_data['cd'])
computer_data['multi']=le.fit_transform(computer_data['multi'])
computer_data['premium']=le.fit_transform(computer_data['premium'])
a=computer_data.columns
#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 
# speed
plt.bar(height = computer_data.speed, x = np.arange(1, 6260, 1))
plt.hist(computer_data.speed) #histogram
plt.boxplot(computer_data.speed) #boxplot

# price
plt.bar(height = computer_data.price, x = np.arange(1, 6260, 1))
plt.hist(computer_data.price) #histogram
plt.boxplot(computer_data.price) #boxplot and found outliers. now ignoring those outliers and proceeding

# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(computer_data.price, dist = "norm", plot = pylab)

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(computer_data.iloc[:, :])
                             
# Correlation matrix 
computer_data.corr()

# we see there exists High collinearity between input variables especially between
# [trend & x]so there exists collinearity problem
# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model

ml1 = smf.ols('price ~ speed + hd + ram + screen + cd + multi + premium + ads + trend', data = computer_data).fit() # regression model
ml1.summary()
# p-values for total datset is leass thanan 0.05

# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1)
# Studentized Residuals = Residual/standard deviation of residuals
# index 83 is showing high influence so we can exclude that entire row

startup_new = computer_data.drop(startup.index[[89])

# Preparing model                  
ml1_new = smf.ols('price ~ speed + hd + ram + screen + cd + multi + premium + ads + trend', data = computer_data).fit()    
ml1_new.summary()

# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables
rsq_price = smf.ols('price ~ speed + hd + ram + screen + cd + multi + premium + ads + trend', data = computer_data).fit().rsquared  
vif_price = 1/(1 - rsq_price) 

rsq_speed = smf.ols('speed ~ price + hd + ram + screen + cd + multi + premium + ads + trend', data = computer_data).fit().rsquared  
vif_speed = 1/(1 - rsq_speed)

rsq_hd = smf.ols('hd ~ price + speed + ram + screen + cd + multi + premium + ads + trend', data = computer_data).fit().rsquared  
vif_hd = 1/(1 - rsq_hd) 

rsq_ram = smf.ols('ram ~ price + speed + hd + screen + cd + multi + premium + ads + trend', data = computer_data).fit().rsquared  
vif_ram = 1/(1 - rsq_ram) 

rsq_screen = smf.ols('screen ~ price + speed + hd + ram + cd + multi + premium + ads + trend', data = computer_data).fit().rsquared  
vif_screen = 1/(1 - rsq_screen) 

rsq_cd = smf.ols('cd ~ price + speed + hd + ram + screen + multi + premium + ads + trend', data = computer_data).fit().rsquared  
vif_cd = 1/(1 - rsq_cd) 

rsq_multi = smf.ols('multi ~ price + speed + hd + ram + screen + cd + premium + ads + trend', data = computer_data).fit().rsquared  
vif_multi = 1/(1 - rsq_multi) 

rsq_premium = smf.ols('premium ~ price + speed + hd + ram + screen + cd + multi + ads + trend', data = computer_data).fit().rsquared  
vif_premium = 1/(1 - rsq_premium) 

rsq_ads = smf.ols('ads ~ price + speed + hd + ram + screen + cd + premium + multi + trend', data = computer_data).fit().rsquared  
vif_ads = 1/(1 - rsq_ads) 

rsq_trend = smf.ols('trend ~ price + speed + hd + ram + screen + cd + premium + ads + multi', data = computer_data).fit().rsquared  
vif_trend = 1/(1 - rsq_trend) 


# Storing vif values in a data frame
d1 = {'Variables':['trend','price','speed', 'hd' , 'ram' , 'screen' , 'cd', 'premium', 'ads', 'multi'], 'VIF':[vif_price, vif_speed, vif_hd, vif_ram,vif_screen ,vif_cd,vif_multi,vif_premium,vif_ads,vif_trend]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# As WT is having highest VIF value, we are going to drop this from the prediction model

# Final model
final_ml = smf.ols('price ~ speed + hd + ram + screen + cd + multi  + ads + trend', data = computer_data).fit()
final_ml.summary() 

# Prediction
pred = final_ml.predict(computer_data)

# Q-Q plot
res = final_ml.resid
sm.qqplot(res)

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

import seaborn as sns
# Residuals vs Fitted plot
sns.residplot(x = pred, y = computer_data.price, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(final_ml)


### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
company_train, company_test = train_test_split(computer_data, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols('price ~ speed + hd + ram + screen + cd + multi + premium + ads + trend', data = company_train).fit()

# prediction on test data set 
test_pred = model_train.predict(company_test)

# test residual values 
test_resid = test_pred - company_test.price
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse


# train_data prediction
train_pred = model_train.predict(company_train)

# train residual values 
train_resid  = train_pred - company_train.price
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse


