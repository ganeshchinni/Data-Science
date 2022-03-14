# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 12:41:12 2021

@author: chinni
"""

import pandas as pd
import numpy as np
time=pd.read_csv(r"C:\data\data science\Study material\Simple Linear Regression\Datasets_SLR\delivery_time.csv")
time.info()
time.columns
#Measures of Central Tendency / First moment business decision
time['Delivery_Time'].mean()
time['Delivery_Time'].median()
time['Sorting_Time'].mean()
time['Sorting_Time'].median()
#Measures of Dispersion / Second moment business decision
time['Delivery_Time'].var()
time['Delivery_Time'].std()
time['Sorting_Time'].var()
time['Sorting_Time'].std()
range1=max(time['Delivery_Time'])-min(time['Delivery_Time'])
range2=max(time['Sorting_Time'])-min(time['Sorting_Time'])
range1,range2
#third moment business decision
time['Delivery_Time'].skew() #from out put is will say positive skew
time['Sorting_Time'].skew() #from out put is will say positive skew
# Fourth moment business decision
time['Delivery_Time'].kurt() #from out put is will say positive so it is Leptokurtic
time['Sorting_Time'].kurt() #from out put is will say negitive value, so it is platykurtic
# Data Visualization
import matplotlib.pyplot as plt
time.shape
plt.bar(height = time['Delivery_Time'], x = np.arange(1, 22, 1))
plt.bar(height = time['Sorting_Time'], x = np.arange(1, 22, 1))
plt.hist(time.Delivery_Time,color='orange')
plt.hist(time.Sorting_Time)
plt.boxplot(time.Weight_gained) # from output found no outliers
plt.boxplot(time.Sorting_Time) # from output found no outliers
#scatter plot
plt.scatter(x=time['Delivery_Time'], y=time['Sorting_Time'],color='maroon')
#cprrelation
np.corrcoef(time['Delivery_Time'],time['Sorting_Time'])
# Covariance
# NumPy does not have a function to calculate the covariance between two variables directly. 
# Function for calculating a covariance matrix called cov() 
# By default, the cov() function will calculate the unbiased or sample covariance between the provided random variables.

cov_output=np.cov((time['Delivery_Time'],time['Sorting_Time']))[0,1]
cov_output

# Import library
import statsmodels.formula.api as smf

# Simple Linear Regression
model = smf.ols('Sorting_Time ~ Delivery_Time', data = time).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(time['Delivery_Time']))

# Regression Line
plt.scatter(time.Delivery_Time, time.Sorting_Time)
plt.plot(time.Delivery_Time, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res1 = time.Sorting_Time - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

######### Model building on Transformed Data
# Log Transformation
# x = log(waist); y = at

plt.scatter(x = np.log(time['Delivery_Time']), y = time['Sorting_Time'], color = 'brown')
np.corrcoef(np.log(time.Delivery_Time), time.Sorting_Time) #correlation

model2 = smf.ols('Sorting_Time ~ np.log(Delivery_Time)', data = time).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(time['Delivery_Time']))

# Regression Line
plt.scatter(np.log(time.Delivery_Time), time.Sorting_Time)
plt.plot(np.log(time.Delivery_Time), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = time.Sorting_Time - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2


#### Exponential transformation
# x = waist; y = log(at)

plt.scatter(x = time['Delivery_Time'], y = np.log(time['Sorting_Time']), color = 'orange')
np.corrcoef(time.Delivery_Time, np.log(time.Sorting_Time)) #correlation

model3 = smf.ols('np.log(Sorting_Time) ~ Delivery_Time', data = time).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(time['Delivery_Time']))
pred3_at = np.exp(pred3)
pred3_at

# Regression Line
plt.scatter(time.Delivery_Time, np.log(time.Sorting_Time))
plt.plot(time.Delivery_Time, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 = time.Sorting_Time - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3


#### Polynomial transformation
# x = waist; x^2 = waist*waist; y = log(at)

model4 = smf.ols('np.log(Sorting_Time) ~ Delivery_Time + I(Delivery_Time*Delivery_Time)', data = time).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(time))
pred4_at = np.exp(pred4)
pred4_at

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = time.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)
# y = time.iloc[:, 1].values


plt.scatter(time.Delivery_Time, np.log(time.Sorting_Time))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error calculation
res4 = time.Sorting_Time - pred4_at
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4


# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)
table_rmse

# The best model

from sklearn.model_selection import train_test_split

train, test = train_test_split(time, test_size = 0.2)

finalmodel = smf.ols('np.log(Sorting_Time) ~ Delivery_Time + I(Delivery_Time*Delivery_Time)', data = train).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_cal = np.exp(test_pred)
pred_test_cal

# Model Evaluation on Test data
test_res = test.Sorting_Time - pred_test_cal
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse


# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_cal = np.exp(train_pred)
pred_train_cal

# Model Evaluation on train data
train_res = train.Sorting_Time - pred_train_cal
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse