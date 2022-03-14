# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 10:12:58 2021

@author: chinni
"""

############Exploratory Data Analysis###############

'''Q1) Calculate Skewness, Kurtosis using R/Python code & draw inferences on the following data.
Hint: [Insights drawn from the data such as data is normally distributed/not, outliers, 
       measures like mean, median, mode, variance, std. deviation]'''

import pandas as pd
data=pd.read_csv(r"C:\data\data science\Study material\Statistical data visualization plots\Statistical Datasets\Statistical Datasets\Q1_a.csv")
data.drop(['Index'],axis=1,inplace=True)#removing unwanted extra index column
import seaborn as sns #data visualization
sns.boxplot(data.speed)#found no outliers
sns.boxplot(data.dist)#found one outlier
data.describe()
pip install feature_engine
from feature_engine.outliers import winsorizer
IQR=data['dist'].quantile(0.75)-data['dist'].quantile(0.25)
lowerLimit=data['dist'].quantile(0.25)-IQR*1.5
upperLimit=data['dist'].quantile(0.75)+IQR*1.5
#trimming method.
import numpy as np
##trimming tech.
outliersdata=np.where(data['dist']>upperLimit,True,np.where(data['dist']<lowerLimit,True,False))
data_trim = data.loc[~(outliersdata)]
sns.boxplot(data_trim.dist)#now no outliers
#or
'''replacing tech
data['data_replaced'] = pd.DataFrame(np.where(data['dist'] > upperLimit, upperLimit, np.where(data['dist'] < lowerLimit, lowerLimit, data['dist'])))
sns.boxplot(data.data_replaced)'''

##calculating 3rd moment business decision
from scipy import stats
data_trim.dist.skew()
#the output in positive value. So it is positive skew
data.speed.skew()
#the output in negative value. So it is negative skew

##calculating 3rd moment business decision
data_trim.dist.kurt()
#the output in negative value. So it is platy kurtic
data.dist.kurt()
#the output in positive value. So it is lepty kurtic



#ans:b

df=pd.read_csv(r"C:\data\data science\Study material\Statistical data visualization plots\Statistical Datasets\Statistical Datasets\Q2_b.csv")
df.info()
df.drop(['Unnamed: 0'],axis=1,inplace=True)# removing unwanted column
import seaborn as sns
sns.boxplot(df.SP)#found outliers 
sns.boxplot(df.WT)#found outliers 
#finding IQR values

IQR=df['SP'].quantile(0.75)-df['SP'].quantile(0.25)
lLimit=df['SP'].quantile(0.25)-IQR*1.5
uLimit=df['SP'].quantile(0.75)+IQR*1.5
df_outliers=np.where(df['SP']>uLimit,True,np.where(df['SP']<lLimit,True,False))
df_trim = df.loc[~(df_outliers)]
sns.boxplot(df_trim.SP)

IQR1=df['WT'].quantile(0.75)-df['SP'].quantile(0.25)
lLimit1=df['WT'].quantile(0.25)-IQR1*1.5
uLimit1=df['WT'].quantile(0.75)+IQR1*1.5
df_outliers1=np.where(df['WT']>uLimit1,True,np.where(df['WT']<lLimit1,True,False))
df_trim1 = df.loc[~(df_outliers1)]
sns.boxplot(df_trim.WT)
##still i found outliers, no removing by replacing tecniqe
df['df_replaced'] = pd.DataFrame(np.where(df['SP'] > uLimit, uLimit, np.where(df['SP'] < lLimit, lLimit, df['SP'])))
sns.boxplot(df.df_replaced)

df['df_replaced1'] = pd.DataFrame(np.where(df['WT'] > uLimit1, uLimit1, np.where(df['WT'] < lLimit1, lLimit1, df['WT'])))
sns.boxplot(df.df_replaced1)

##calculating 3rd moment business decision
from scipy import stats
df.df_replaced.skew()
#the output in positive value. So it is positive skew
df.df_replaced1.skew()
#the output is 0 value. So it is symmetric

##calculating 3rd moment business decision
df.df_replaced.kurt()
#the output  in positive value. So it is lepty kurtic
df.df_replaced1.kurt()
#the output in 0 value. So it is meso kurtic

