# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 16:47:38 2021

@author: chinni
"""

'''Perform hierarchical and K-means clustering on the dataset.
    After that, perform PCA on the dataset and extract the first
    3 principal components and make a new dataset with these 3 principal
    components as the columns. Now, on this new dataset, perform hierarchical
    and K-means clustering. Compare the results of clustering on the original datase
    and clustering on the principal components dataset (use the scree plot technique to
    obtain the optimum number of clustersin K-means clustering and check if you’re getting similarr
    esults with and without PCA).'''
    
    
import pandas as pd
data=pd.read_csv(r"C:\data\data science\Study material\Data mining unsupervised learning-(pca)\Datasets_PCA\wine.csv")
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
plt.figure(figsize=(55,30));plt.title('Hierarchical clustering dendogram');plt.xlabel("index");plt.ylabel("distance")
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

final_ax = final.plot(x='comp0', y='Alcohol', kind='scatter',figsize=(12,8)) # plotting graph with pca values and original values
final_ax = final.plot(y='comp0', x='Alcohol', kind='scatter',figsize=(12,8)) # plotting graph with pca values and original values

####for this final data performing hierarchical and Kmeans clustering


#hierarchical clustering
#for creating dendogram
z1=sch.linkage(final,method="complete",metric="euclidean")
#dendogram
plt.figure(figsize=(45,15));plt.title('Hierarchical clustering dendogram');plt.xlabel("index");plt.ylabel("distance")
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
df_xy1.plot(x="X1",y="Y1",c=model2.labels_,kind="scatter",s=10,cmap=plt.cm.coolwarm)


