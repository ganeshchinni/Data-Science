# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 12:46:31 2021

@author: chinni
"""

'''5.Perform clustering on mixed data. Convert the categorical variables to numeric by using dummies or label encoding
    and perform normalization techniques. The dataset has the details of customers related to their auto insurance.
    Refer to Autoinsurance.csv dataset.'''


import pandas as pd
insurance=pd.read_csv(r"C:\data\data science\Study material\Data mining unsupervised learning (K-means)\Datasets_Kmeans\AutoInsurance (1).csv")
insurance.isna().sum()
insurance.info()
a=insurance.columns
# Label Encoder
from sklearn.preprocessing import LabelEncoder
# creating instance of labelencoder
labelencoder = LabelEncoder()
insurance=insurance.drop(['Customer'],axis=1)
###label encoding

insurance['State']= labelencoder.fit_transform(insurance['State'])
insurance['Customer Lifetime Value'] = labelencoder.fit_transform(insurance['Customer Lifetime Value'])
insurance['Response']= labelencoder.fit_transform(insurance['Response'])
insurance['Coverage']= labelencoder.fit_transform(insurance['Coverage'])
insurance['Education']= labelencoder.fit_transform(insurance['Education'])
insurance['Effective To Date']= labelencoder.fit_transform(insurance['Effective To Date'])
insurance['EmploymentStatus']= labelencoder.fit_transform(insurance['EmploymentStatus'])
insurance['Response']= labelencoder.fit_transform(insurance['Response'])
insurance['Gender']= labelencoder.fit_transform(insurance['Gender'])
insurance['Location Code']= labelencoder.fit_transform(insurance['Location Code'])
insurance['Marital Status']= labelencoder.fit_transform(insurance['Marital Status'])
insurance['Policy Type']= labelencoder.fit_transform(insurance['Policy Type'])
insurance['Policy']= labelencoder.fit_transform(insurance['Policy'])
insurance['Renew Offer Type']= labelencoder.fit_transform(insurance['Renew Offer Type'])
insurance['Sales Channel']= labelencoder.fit_transform(insurance['Sales Channel'])
insurance['Vehicle Class']= labelencoder.fit_transform(insurance['Vehicle Class'])
insurance['Vehicle Size']= labelencoder.fit_transform(insurance['Vehicle Size'])
insurance.info()
insurance.describe()
##min and max values have large variation. so proceeding with normalizing
#normalization function
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return(x)
#normalized data frame
insurance_norm=norm_func(insurance)
res_nrn=insurance_norm.describe()
from sklearn.cluster import KMeans
#generating random uniform numbers
import numpy as np
x=np.random.uniform(0,1,50)
y=np.random.uniform(0,1,50)
df_xy=pd.DataFrame(columns=["X","Y"])
df_xy.X=x
df_xy.Y=y
df_xy.plot(x="X",y="Y", kind="scatter")
model1=KMeans(n_clusters=4).fit(df_xy)
import matplotlib.pyplot as plt
df_xy.plot(x="X",y="Y",c=model1.labels_,kind="scatter",s=10,cmap=plt.cm.coolwarm)
