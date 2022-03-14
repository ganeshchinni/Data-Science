# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 12:41:12 2021

@author: chinni
"""
import pandas as pd
import numpy as np
calories=pd.read_csv(r"C:\data\data science\Study material\Simple Linear Regression\Datasets_SLR\calories_consumed.csv")
calories.info()
calories.columns
calories.columns = ['Weight_gained', 'Calories_Consumed']
#Measures of Central Tendency / First moment business decision
calories['Weight_gained'].mean()
calories['Weight_gained'].median()
calories['Calories_Consumed'].mean()
calories['Calories_Consumed'].median()
#Measures of Dispersion / Second moment business decision
calories['Weight_gained'].var()
calories['Weight_gained'].std()
calories['Calories_Consumed'].var()
calories['Calories_Consumed'].std()
range1=max(calories['Weight_gained'])-min(calories['Weight_gained'])
range2=max(calories['Calories_Consumed'])-min(calories['Calories_Consumed'])
range1,range2
#third moment business decision
calories['Weight_gained'].skew() #from out put is will say positive skew
calories['Calories_Consumed'].skew() #from out put is will say positive skew
# Fourth moment business decision
calories['Weight_gained'].kurt() #from out put is will say positive so it is Leptokurtic
calories['Calories_Consumed'].kurt() #from out put is will say negitive value, so it is platykurtic
# Data Visualization
import matplotlib.pyplot as plt
calories.shape
plt.bar(height = calories['Weight_gained'], x = np.arange(1, 15, 1))
plt.bar(height = calories['Calories_Consumed'], x = np.arange(1, 15, 1))
plt.hist(calories.Weight_gained,color='orange')
plt.hist(calories.Calories_Consumed)
plt.boxplot(calories.Weight_gained) # from output found no outliers
plt.boxplot(calories.Calories_Consumed) # from output found no outliers
#scatter plot
plt.scatter(x=calories['Weight_gained'], y=calories['Calories_Consumed'],color='maroon')
#cprrelation
np.corrcoef(calories['Weight_gained'],calories['Calories_Consumed'])
# Covariance
# NumPy does not have a function to calculate the covariance between two variables directly. 
# Function for calculating a covariance matrix called cov() 
# By default, the cov() function will calculate the unbiased or sample covariance between the provided random variables.

cov_output=np.cov((calories['Weight_gained'],calories['Calories_Consumed']))[0,1]
cov_output

# Import library
import statsmodels.formula.api as smf

# Simple Linear Regression
model = smf.ols('Calories_Consumed ~ Weight_gained', data = calories).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(calories['Weight_gained']))

# Regression Line
plt.scatter(calories.Weight_gained, calories.Calories_Consumed)
plt.plot(calories.Weight_gained, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res1 = calories.Calories_Consumed - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

######### Model building on Transformed Data
# Log Transformation
# x = log(waist); y = at

plt.scatter(x = np.log(calories['Weight_gained']), y = calories['Calories_Consumed'], color = 'brown')
np.corrcoef(np.log(calories.Weight_gained), calories.Calories_Consumed) #correlation

model2 = smf.ols('Calories_Consumed ~ np.log(Weight_gained)', data = calories).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(calories['Weight_gained']))

# Regression Line
plt.scatter(np.log(calories.Weight_gained), calories.Calories_Consumed)
plt.plot(np.log(calories.Weight_gained), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = calories.Calories_Consumed - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2


#### Exponential transformation
# x = waist; y = log(at)

plt.scatter(x = calories['Weight_gained'], y = np.log(calories['Calories_Consumed']), color = 'orange')
np.corrcoef(calories.Weight_gained, np.log(calories.Calories_Consumed)) #correlation

model3 = smf.ols('np.log(Calories_Consumed) ~ Weight_gained', data = calories).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(calories['Weight_gained']))
pred3_at = np.exp(pred3)
pred3_at

# Regression Line
plt.scatter(calories.Weight_gained, np.log(calories.Calories_Consumed))
plt.plot(calories.Weight_gained, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 = calories.Calories_Consumed - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3


#### Polynomial transformation
# x = waist; x^2 = waist*waist; y = log(at)

model4 = smf.ols('np.log(Calories_Consumed) ~ Weight_gained + I(Weight_gained*Weight_gained)', data = calories).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(calories))
pred4_at = np.exp(pred4)
pred4_at

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = calories.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)
# y = calories.iloc[:, 1].values


plt.scatter(calories.Weight_gained, np.log(calories.Calories_Consumed))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error calculation
res4 = calories.Calories_Consumed - pred4_at
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

train, test = train_test_split(calories, test_size = 0.2)

finalmodel = smf.ols('np.log(Calories_Consumed) ~ Weight_gained + I(Weight_gained*Weight_gained)', data = train).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_cal = np.exp(test_pred)
pred_test_cal

# Model Evaluation on Test data
test_res = test.Calories_Consumed - pred_test_cal
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse


# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_cal = np.exp(train_pred)
pred_train_cal

# Model Evaluation on train data
train_res = train.Calories_Consumed - pred_train_cal
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse



