# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 00:06:23 2021

@author: chinni
"""

'''4.Perform clustering analysis on the telecom dataset. The data is a mixture of both categorical
    and numerical data. It consists of the number of customers who churn. Derive insights and get
    possible information on factors that may affect the churn decision. Refer to Telco_customer_churn.xlsx dataset.'''
    
import pandas as pd
import numpy as np
telecom=pd.read_excel(r"C:\data\data science\Study material\Data mining unsupervised learning (K-means)\Datasets_Kmeans\Telco_customer_churn (1).xlsx")
telecom1.info()
telecom. isna().sum()
telecom1=telecom.drop(['Customer ID'],axis=1)
telecom1=pd.get_dummies(telecom,drop_first = True)
telecom1.describe()# min and max are having too much variation
#normalization function
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return(x)
#normalized data frame
telecom1_norm=norm_func(telecom1)
res_nrn=telecom1_norm.describe()
from sklearn.cluster import KMeans
#generating random uniform numbers
x=np.random.uniform(0,1,50)
y=np.random.uniform(0,1,50)
df_xy=pd.DataFrame(columns=["X","Y"])
df_xy.X=x
df_xy.Y=y
df_xy.plot(x="X",y="Y", kind="scatter")
model1=KMeans(n_clusters=4).fit(df_xy)
model1=pd.DataFrame(model1)
import matplotlib.pyplot as plt
df_xy.plot(x="X",y="Y",c=model1.labels_,kind="scatter",s=10,cmap=plt.cm.coolwarm)


