# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 13:46:52 2021

@author: chinni
"""

'''2.	Perform clustering for the crime data and identify the number of clusters 
    formed and draw inferences. Refer to crime_data.csv dataset.'''
    
import pandas as pd
import numpy as np
crim=pd.read_csv(r"C:\data\data science\Study material\Data mining unsupervised learning & hierarchical clustering\Dataset_Assignment Clustering\crime_data.csv")
crim.info()
crim.columns
plt.boxplot(crim.Murder)
plt.boxplot(crim.Assault)
plt.boxplot(crim.UrbanPop)
plt.boxplot(crim.Rape)
#found one outlier in Rape column. going to ignore
#finding IQR limits
# Detection of outliers (find limits for crim based on IQR)
IQR = crim['Rape'].quantile(0.75) - crim['Rape'].quantile(0.25)
lower_limit = crim['Rape'].quantile(0.25) - (IQR * 1.5)
upper_limit = crim['Rape'].quantile(0.75) + (IQR * 1.5)
############### 1. Remove (let's replace the dataset) ################
crim['Rape'] = pd.DataFrame(np.where(crim['Rape'] > upper_limit, upper_limit, np.where(crim['Rape'] < lower_limit, lower_limit, crim['Rape'])))
plt.boxplot(crim['Rape'])
crim1=crim.drop(['Unnamed: 0'],axis=1)
crim1.describe()# min and max are having too much variation
#normalization function
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return(x)
#normalized data frame
crim_norm=norm_func(crim1)
res_nrn=crim_norm.describe()

#standardization 
from sklearn.preprocessing import StandardScaler
#initialization of scalar
scalar=StandardScaler()
#scaling data
crim_std=scalar.fit_transform(crim1)
#converting array back to data frame
dataset=pd.DataFrame(crim_std)
res_std=dataset.describe()

#for creating dendogram
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
z=linkage(crim_norm, method = "complete", metric = "euclidean")
# dendogram
import matplotlib.pyplot as plt
plt.figure(figsize=(20, 10));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 45,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()

