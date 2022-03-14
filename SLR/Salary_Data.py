# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 10:20:05 2021

@author: chinni
"""

import pandas as pd   
import numpy as np 
import matplotlib.pyplot as plt

#Loading the data set into python
salary_data = pd.read_csv(r"C:\data\data science\Study material\Simple Linear Regression\Datasets_SLR\Salary_Data.csv")
salary_data.describe()
#Measures of Central Tendency / First moment business decision
salary_data['YearsExperience'].mean()
salary_data['YearsExperience'].median()
salary_data['Salary'].mean()
salary_data['Salary'].median()
#Measures of Dispersion / Second moment business decision
salary_data['YearsExperience'].var()
salary_data['YearsExperience'].std()
salary_data['Salary'].var()
salary_data['Salary'].std()
range1=max(salary_data['YearsExperience'])-min(salary_data['YearsExperience'])
range2=max(salary_data['Salary'])-min(salary_data['Salary'])
range1,range2
#third moment business decision
salary_data['YearsExperience'].skew() #from out put is will say positive skew
salary_data['Salary'].skew() #from out put is will say positive skew
# Fourth moment business decision
salary_data['YearsExperience'].kurt() #from out put is will say positive so it is Leptokurtic
salary_data['Salary'].kurt() #from out put is will say negitive value, so it is platykurtic
# Data Visualization
import matplotlib.pyplot as plt
salary_data.shape
plt.bar(height = salary_data['YearsExperience'], x = np.arange(1, 31, 1))
plt.bar(height = salary_data['Salary'], x = np.arange(1, 31, 1))
plt.hist(salary_data.YearsExperience,color='orange')
plt.hist(salary_data.Salary)
plt.boxplot(salary_data.YearsExperience) # from output found no outliers
plt.boxplot(salary_data.Salary) # from output found no outliers
#scatter plot
plt.scatter(x=salary_data['YearsExperience'], y=salary_data['Salary'],color='maroon')
#cprrelation
np.corrcoef(salary_data['YearsExperience'],salary_data['Salary'])
# Covariance
# NumPy does not have a function to calculate the covariance between two variables directly. 
# Function for calculating a covariance matrix called cov() 
# By default, the cov() function will calculate the unbiased or sample covariance between the provided random variables.

cov_output=np.cov((salary_data['YearsExperience'],salary_data['Salary']))[0,1]
cov_output

#Simple linear regression
import statsmodels.formula.api as smf
#Generating the model
model = smf.ols('Salary ~ YearsExperience', data = salary_data).fit()
model.summary()

prediction = model.predict(pd.DataFrame(salary_data['YearsExperience']))

# Scatter plot and the regression line
plt.scatter(salary_data.YearsExperience, salary_data.Salary)
plt.plot(salary_data.YearsExperience, prediction, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Calculating the error
result = salary_data.Salary - prediction
res_sqr = result * result
mse = np.mean(res_sqr)
RMSE = np.sqrt(mse)
RMSE

#Applying transformation techniques to prune the model
#Applying logarthemic transformation
plt.scatter(x = np.log(salary_data['YearsExperience']), y = salary_data['Salary'], color = 'brown')
#Correlation
np.corrcoef(np.log(salary_data.YearsExperience), salary_data.Salary) 

model2 = smf.ols('Salary ~ np.log(YearsExperience)', data = salary_data).fit()
model2.summary()

prediction2 = model2.predict(pd.DataFrame(salary_data['YearsExperience']))

# Scatter plot and the regression line
plt.scatter(np.log(salary_data.YearsExperience), salary_data.Salary)
plt.plot(np.log(salary_data.YearsExperience), prediction2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Calculating the error
result2 = salary_data.Salary - prediction2
res_sqr2 = result2 * result2
mse2 = np.mean(res_sqr2)
RMSE2 = np.sqrt(mse2)
RMSE2

#Applying Exponentian Technique
plt.scatter(x =  salary_data['YearsExperience'], y = np.log(salary_data['Salary']), color = 'orange')
np.corrcoef(salary_data.YearsExperience, np.log(salary_data.Salary))

model3 = smf.ols('np.log(Salary) ~ YearsExperience', data = salary_data).fit()
model3.summary()

prediction3 = model3.predict(pd.DataFrame(salary_data['YearsExperience']))
pred3_at = np.exp(prediction3)
pred3_at

# Scatter plot and the regression line
plt.scatter(salary_data.YearsExperience, np.log(salary_data.Salary))
plt.plot(salary_data.YearsExperience, prediction3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

#Calculating the error
result3 = salary_data.Salary - pred3_at
res_sqr3 = result3 * result3
mse3 = np.mean(res_sqr3)
RMSE3 = np.sqrt(mse3)
RMSE3

#Applying polynomial transformation technique
model4 = smf.ols('np.log(Salary) ~ YearsExperience + I(YearsExperience*YearsExperience)', data = salary_data).fit()
model4.summary()

prediction4 = model4.predict(pd.DataFrame(salary_data))
pred4_at = np.exp(prediction4)
pred4_at

#Scatter plot and the regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = salary_data.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)



plt.scatter(salary_data.YearsExperience, np.log(salary_data.Salary))
plt.plot(X, prediction4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Calculating the error
result4 = salary_data.Salary - pred4_at
res_sqr4 = result4 * result4
mse4 = np.mean(res_sqr4)
RMSE4 = np.sqrt(mse4)
RMSE4


# Choosing the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([RMSE, RMSE2, RMSE3, RMSE4])}
table_rmse = pd.DataFrame(data)
table_rmse

#Selecting the best model for the final modeling of the data set
from sklearn.model_selection import train_test_split

train, test = train_test_split(salary_data, test_size = 0.2)

finalmodel = smf.ols('np.log(Salary) ~ YearsExperience + I(YearsExperience*YearsExperience)', data = salary_data).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_Salary = np.exp(test_pred)
pred_test_Salary

# Model Evaluation on Test data
test_res = test.Salary - pred_test_Salary
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse


# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_Salary = np.exp(train_pred)
pred_train_Salary

# Model Evaluation on train data
train_res = train.Salary - pred_train_Salary
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse
