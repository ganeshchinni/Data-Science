# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 10:40:58 2021

@author: chinni
"""


import pandas as pd   
import numpy as np 
import matplotlib.pyplot as plt

#Loading the data set into python
sat_data  = pd.read_csv(r"C:\data\data science\Study material\Simple Linear Regression\Datasets_SLR\SAT_GPA.csv")
sat_data.describe()
#Measures of Central Tendency / First moment business decision
sat_data['SAT_Scores'].mean()
sat_data['SAT_Scores'].median()
sat_data['GPA'].mean()
sat_data['GPA'].median()
#Measures of Dispersion / Second moment business decision
sat_data['SAT_Scores'].var()
sat_data['SAT_Scores'].std()
sat_data['GPA'].var()
sat_data['GPA'].std()
range1=max(sat_data['SAT_Scores'])-min(sat_data['SAT_Scores'])
range2=max(sat_data['GPA'])-min(sat_data['GPA'])
range1,range2
#third moment business decision
sat_data['SAT_Scores'].skew() #from out put is will say positive skew
sat_data['GPA'].skew() #from out put is will say positive skew
# Fourth moment business decision
sat_data['SAT_Scores'].kurt() #from out put is will say negitive value, so it is platykurtic
sat_data['GPA'].kurt() #from out put is will say negitive value, so it is platykurtic
# Data Visualization
import matplotlib.pyplot as plt
sat_data.shape
plt.bar(height = sat_data['SAT_Scores'], x = np.arange(1, 201, 1))
plt.bar(height = sat_data['GPA'], x = np.arange(1, 201, 1))
plt.hist(sat_data.SAT_Scores,color='orange')
plt.hist(sat_data.GPA)
plt.boxplot(sat_data.SAT_Scores) # from output found no outliers
plt.boxplot(sat_data.GPA) # from output found no outliers
#scatter plot
plt.scatter(x=sat_data['SAT_Scores'], y=sat_data['GPA'],color='maroon')
#cprrelation
np.corrcoef(sat_data['SAT_Scores'],sat_data['GPA'])
# Covariance
# NumPy does not have a function to calculate the covariance between two variables directly. 
# Function for calculating a covariance matrix called cov() 
# By default, the cov() function will calculate the unbiased or sample covariance between the provided random variables.

cov_output=np.cov((sat_data['SAT_Scores'],sat_data['GPA']))[0,1]
cov_output

#Simple linear regression
import statsmodels.formula.api as smf
#Generating the model
model = smf.ols('SAT_Scores ~ GPA', data = sat_data).fit()
model.summary()

prediction = model.predict(pd.DataFrame(sat_data['GPA']))

# Scatter plot and the regression line
plt.scatter(sat_data.SAT_Scores, sat_data.GPA)
plt.plot(sat_data.SAT_Scores, sat_data.GPA, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Calculating the error
result = sat_data.SAT_Scores - prediction
res_sqr = result * result
mse = np.mean(res_sqr)
RMSE = np.sqrt(mse)
RMSE

#Applying transformation techniques to prune the model
#Applying logarthemic transformation
plt.scatter(x = np.log(sat_data['GPA']), y = sat_data['SAT_Scores'], color = 'brown')
#Correlation
np.corrcoef(np.log(sat_data.GPA), sat_data.SAT_Scores) 

model2 = smf.ols('SAT_Scores ~ np.log(GPA)', data = sat_data).fit()
model2.summary()

prediction2 = model2.predict(pd.DataFrame(sat_data['GPA']))

# Scatter plot and the regression line
plt.scatter(np.log(sat_data.GPA), sat_data.SAT_Scores)
plt.plot(np.log(sat_data.GPA), prediction2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Calculating the error
result2 = sat_data.SAT_Scores - prediction2
res_sqr2 = result2 * result2
mse2 = np.mean(res_sqr2)
RMSE2 = np.sqrt(mse2)
RMSE2

#Applying Exponentian Technique
plt.scatter(x =  sat_data['GPA'], y = np.log(sat_data['SAT_Scores']), color = 'orange')
np.corrcoef(sat_data.GPA, np.log(sat_data.SAT_Scores))

model3 = smf.ols('np.log(SAT_Scores) ~ GPA', data = sat_data).fit()
model3.summary()

prediction3 = model3.predict(pd.DataFrame(sat_data['GPA']))
pred3_at = np.exp(prediction3)
pred3_at

# Scatter plot and the regression line
plt.scatter(sat_data.GPA, np.log(sat_data.SAT_Scores))
plt.plot(sat_data.GPA, prediction3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

#Calculating the error
result3 = sat_data.SAT_Scores - pred3_at
res_sqr3 = result3 * result3
mse3 = np.mean(res_sqr3)
RMSE3 = np.sqrt(mse3)
RMSE3

#Applying polynomial transformation technique
model4 = smf.ols('np.log(SAT_Scores) ~ GPA + I(GPA*GPA)', data = sat_data).fit()
model4.summary()

prediction4 = model4.predict(pd.DataFrame(sat_data))
pred4_at = np.exp(prediction4)
pred4_at

#Scatter plot and the regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = sat_data.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)



plt.scatter(sat_data.GPA, np.log(sat_data.SAT_Scores))
plt.plot(X, prediction4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Calculating the error
result4 = sat_data.SAT_Scores - pred4_at
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

train, test = train_test_split(sat_data, test_size = 0.2)

finalmodel = smf.ols('np.log(SAT_Scores) ~ GPA + I(GPA*GPA)', data = sat_data).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_SAT_Scores = np.exp(test_pred)
pred_test_SAT_Scores

# Model Evaluation on Test data
test_res = test.SAT_Scores - pred_test_SAT_Scores
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse


# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_SAT_Scores = np.exp(train_pred)
pred_train_SAT_Scores

# Model Evaluation on train data
train_res = train.SAT_Scores - pred_train_SAT_Scores
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse
