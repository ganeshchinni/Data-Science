# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 20:46:09 2021

@author: chinni
"""

''' pharmaceuticals manufacturing company is conducting a study on a new medicine to treat heart
    diseases. The company has gathered data from its secondary sources and would like you to provide
    high level analytical insights on the data. Its aim is to segregate patients depending on their age
    group and other factors given in the data. Perform PCA and clustering algorithms on the dataset and
    check if the clusters formed before and after PCA are the same and provide a brief report on your model.
    You can also explore more ways to improve your model.'''
    
    
import pandas as pd
data=pd.read_csv(r"C:\data\data science\Study material\Data mining unsupervised learning-(pca)\Datasets_PCA\heart disease.csv")
data.info()
data.describe()
#normalization function
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return(x)
df_nrm=norm_func(data)
df_nrm.describe()

#hierarchical clustering
#for creating dendogram
import scipy.cluster.hierarchy as sch
z=sch.linkage(df_nrm,method="complete",metric="euclidean")
#dendogram
import matplotlib.pyplot as plt
plt.figure(figsize=(35,20));plt.title('Hierarchical clustering dendogram');plt.xlabel("index");plt.ylabel("distance")
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()

#K-means clustering
from sklearn.cluster import KMeans
import numpy as np
#generating random uniform numbers
x=np.random.uniform(0,1,50)
y=np.random.uniform(0,1,50)
df_xy=pd.DataFrame(columns=["X","Y"])
df_xy.X=x
df_xy.Y=y
df_xy.plot(x="X", y="Y",kind="scatter")
model1=KMeans(n_clusters=3).fit(df_xy)
df_xy.plot(x="X",y="Y",c=model1.labels_,kind="scatter",s=10,cmap=plt.cm.coolwarm)


##PCA (principal component analysis)
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
#normalizing the numericl data
uni_norm=scale(data)
uni_norm
pca=PCA(n_components=13)
pca_values=pca.fit_transform(uni_norm)

#the amount of variance that each PCA explains is

var=pca.explained_variance_ratio_
#PCA weights
pca.components_
pca.components_[0]
var
#comulative variance
var1=np.cumsum(np.round(var,decimals=4)*100)
var1
#variance plot for pca components obtained
plt.plot(var1,color='blue')
#pca scores
pca_values
pca_data = pd.DataFrame(pca_values)
pca_data.columns = "comp0", "comp1", "comp2", "comp3", "comp4", "comp5","comp6","comp7","comp8","comp9","comp10","comp11","comp12"
extract3columns=pca_data.iloc[:,0:3]##extracting 3 columns to merg with original data
final = pd.concat([data, extract3columns], join = 'outer', axis = 1)# adding extracted columns to original data

# Scatter diagram
import matplotlib.pylab as plt
ax = pca_data.plot(x='comp0', y='comp1', kind='scatter',figsize=(12,8))

final_ax = final.plot(x='comp0', y='age', kind='scatter',figsize=(12,8)) # plotting graph with pca values and original values

####for this final data performing hierarchical and Kmeans clustering


#hierarchical clustering
#for creating dendogram
z1=sch.linkage(final,method="complete",metric="euclidean")
#dendogram
plt.figure(figsize=(30,15));plt.title('Hierarchical clustering dendogram');plt.xlabel("index");plt.ylabel("distance")
sch.dendrogram(z1, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()

#K-means clustering
#generating random uniform numbers
x=np.random.uniform(0,1,50)
y=np.random.uniform(0,1,50)
df_xy1=pd.DataFrame(columns=["X1","Y1"])
df_xy1.X1=x
df_xy1.Y1=y
df_xy1.plot(x="X1", y="Y1",kind="scatter")
model1=KMeans(n_clusters=1).fit(df_xy1)
df_xy1.plot(x="X1",y="Y1",c=model1.labels_,kind="scatter",s=10,cmap=plt.cm.coolwarm)


