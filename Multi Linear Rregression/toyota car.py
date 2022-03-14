# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 21:23:14 2021

@author: chinni
"""

import pandas as pd
import numpy as np

# loading the data
toyota = pd.read_csv(r"C:\data\data science\Study material\Multi Linear aregression\Datasets_MLR\ToyotaCorolla.csv")
toyota.drop(toyota.columns.difference(['Price','Age_08_04', 'KM','HP','cc','Doors','Gears','Quarterly_Tax','Weight']), 1, inplace=True)
toyota.describe()
#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

# Price
plt.bar(height = toyota.Price, x = np.arange(1, 1437, 1))
plt.hist(toyota.Price) #histogram
plt.boxplot(toyota.Price) #boxplot

# KM
plt.bar(height = toyota.KM, x = np.arange(1, 1437, 1))
plt.hist(toyota.KM) #histogram
plt.boxplot(toyota.KM) #boxplot

# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(toyota.Price, dist = "norm", plot = pylab)
plt.show()

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(toyota.iloc[:, :])
                             
# Correlation matrix 
toyota.corr()

# we see there exists High collinearity between input variables especially between
# [KM & price],so there exists collinearity problem

# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
ml1 = smf.ols('Price ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight', data = toyota).fit() # regression model

# Summary
ml1.summary()
# p-values for cc, doors are more than 0.05
# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1)
# Studentized Residuals = Residual/standard deviation of residuals
# index 76 is showing high influence so we can exclude that entire row
toyota_new = toyota.drop(toyota.index[[80]])

# Preparing model                  
ml_new = smf.ols('Price ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight', data = toyota_new).fit()    
# Summary
ml_new.summary()

# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables
rsq_Price = smf.ols('Price ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight', data = toyota).fit().rsquared  
vif_Price = 1/(1 - rsq_Price) 

rsq_Age = smf.ols('Age_08_04 ~ Price + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight', data = toyota).fit().rsquared  
vif_Age = 1/(1 - rsq_Age)

rsq_KM = smf.ols('KM ~ Age_08_04 + Price + HP + cc + Doors + Gears + Quarterly_Tax + Weight', data = toyota).fit().rsquared  
vif_KM = 1/(1 - rsq_KM) 

rsq_HP = smf.ols('HP ~ Age_08_04 + KM + Price + cc + Doors + Gears + Quarterly_Tax + Weight', data = toyota).fit().rsquared  
vif_HP = 1/(1 - rsq_HP) 

rsq_cc = smf.ols('cc ~ Age_08_04 + KM + HP + Price + Doors + Gears + Quarterly_Tax + Weight', data = toyota).fit().rsquared  
vif_cc = 1/(1 - rsq_cc) 

rsq_Doors = smf.ols('Doors ~ Age_08_04 + KM + HP + cc + Price + Gears + Quarterly_Tax + Weight', data = toyota).fit().rsquared  
vif_Doors = 1/(1 - rsq_Doors) 

rsq_Gears = smf.ols('Gears ~ Age_08_04 + KM + HP + cc + Doors + Price + Quarterly_Tax + Weight', data = toyota).fit().rsquared  
vif_Gears = 1/(1 - rsq_Gears) 

rsq_quaterly = smf.ols('Quarterly_Tax ~ Age_08_04 + KM + HP + cc + Doors + Gears + Price + Weight', data = toyota).fit().rsquared  
vif_quaterly = 1/(1 - rsq_quaterly) 

rsq_weight = smf.ols('Weight ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Price', data = toyota).fit().rsquared  
vif_weight = 1/(1 - rsq_weight) 


# Storing vif values in a data frame
d1 = {'Variables':['Price','Age_08_04', 'KM','HP','cc','Doors','Gears','Quarterly_Tax','Weight'], 'VIF':[vif_Price, vif_Age, vif_KM, vif_HP,vif_cc, vif_Doors,vif_Gears ,vif_quaterly,vif_weight]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame

# As WT is having highest VIF value, we are going to drop this from the prediction model

# Final model
final_ml = smf.ols('Price ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight', data = toyota).fit()
final_ml.summary() 

# Prediction
pred = final_ml.predict(toyota)

# Q-Q plot
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

import seaborn as sns
# Residuals vs Fitted plot
sns.residplot(x = pred, y = toyota.Price, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()
sm.graphics.influence_plot(final_ml)



