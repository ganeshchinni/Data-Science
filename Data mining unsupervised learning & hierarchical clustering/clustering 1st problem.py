# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 10:21:46 2021

@author: chinni
"""

'''1.Perform clustering for the airlines data to obtain optimum number of clusters. Draw the inferences from the clusters obtained.
    Refer to EastWestAirlines.xlsx dataset.'''
    
import pandas as pd
airlines1=pd.read_excel(r"C:\data\data science\Study material\Data mining unsupervised learning & hierarchical clustering\Dataset_Assignment Clustering\EastWestAirlines.xlsx")
airlines1.info()
airlines=airlines1.drop(['ID#'],axis=1)
airlines.describe()
#normalization function
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return(x)
#normalized data frame
air_norm=norm_func(airlines)
air_norm.describe()

#standardization 
from sklearn.preprocessing import StandardScaler
#initialization of scalar
scalar=StandardScaler()
#scaling data
air_std=scalar.fit_transform(airlines)
#converting array back to data frame
dataset=pd.DataFrame(air_std)
res=dataset.describe()

#for creating dendogram
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
z=linkage(air_norm, method = "complete", metric = "euclidean")
# dendogram
import matplotlib.pyplot as plt

plt.figure(figsize=(40, 25));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 5 # font size for the x axis labels
)
plt.show()
