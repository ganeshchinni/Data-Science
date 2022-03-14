# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 14:59:35 2021

@author: chinni
"""

'''3.Perform clustering analysis on the telecom data set. The data is a mixture of both
    categorical and numerical data. It consists of the number of customers who churn out.
    Derive insights and get possible information on factors that may affect the churn decision.
    Refer to Telco_customer_churn.xlsx dataset.'''
    
import pandas as pd
import numpy as np
telecom=pd.read_excel(r"C:\data\data science\Study material\Data mining unsupervised learning & hierarchical clustering\Dataset_Assignment Clustering\Telco_customer_churn.xlsx")
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

#standardization 
from sklearn.preprocessing import StandardScaler
#initialization of scalar
scalar=StandardScaler()
#scaling data
telecom1_std=scalar.fit_transform(telecom1)
#converting array back to data frame
dataset=pd.DataFrame(telecom1_std)
res_std=dataset.describe()

#for creating dendogram
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
z=linkage(telecom1_norm, method = "complete", metric = "euclidean")
# dendogram
import matplotlib.pyplot as plt
plt.figure(figsize=(25,15));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 45,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()