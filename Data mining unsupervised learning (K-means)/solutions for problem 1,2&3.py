# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 22:26:04 2021

@author: chinni
"""

'''1.Perform K means clustering on the airlines dataset to obtain optimum number of clusters.
    Draw the inferences from the clusters obtained. Refer to EastWestAirlines.xlsx dataset.'''
    
    
import pandas as pd
import numpy as np
import matplotlib.pylab as ptl
airlines=pd.read_excel(r"C:\data\data science\Study material\Data mining unsupervised learning (K-means)\Datasets_Kmeans\EastWestAirlines (1).xlsx")
from sklearn.cluster import KMeans
#generating random uniform  numbers
x=np.random.uniform(0,1,50)
y=np.random.uniform(0,1,50)
df_xy=pd.DataFrame(columns=["x","y"])
df_xy.x=x
df_xy.y=y
df_xy.plot(x="x",y="y",kind='scatter')
model1=KMeans(n_clusters=3).fit(df_xy)
df_xy.plot(x="x",y="y",c=model1.labels_,kind="scatter", s=10,cmap=ptl.cm.coolwarm)


'''2.Perform clustering for the crime data and identify the number of clusters
    formed and draw inferences. Refer to crime_data.csv dataset.'''
    
import pandas as pd
import numpy as np
import matplotlib.pylab as ptl
data=pd.read_csv(r"C:\data\data science\Study material\Data mining unsupervised learning (K-means)\Datasets_Kmeans\crime_data (1).csv")
data.columns
data.drop(['Unnamed: 0'],axis=1,inplace=True)
from sklearn.cluster import KMeans
#generating random uniform numbers
a=np.random.uniform(0,1,60)
b=np.random.uniform(0,1,60)
df_ab=pd.DataFrame(columns=['A','B'])
df_ab.A=a
df_ab.B=b
df_ab.plot(x='A',y='B',kind="scatter")
model2=KMeans(n_clusters=5).fit(df_ab)
df_ab.plot(x="A",y="B",c=model2.labels_,kind="scatter",s=20,cmap=ptl.cm.coolwarm)


'''3.Analyze the information given in the following ‘Insurance Policy dataset’
    to create clusters of persons falling in the same type. Refer to Insurance Dataset.csv'''
    
import pandas as pd
import numpy as np
import matplotlib.pylab as ptl
insurence=pd.read_csv(r"C:\data\data science\Study material\Data mining unsupervised learning (K-means)\Datasets_Kmeans\Insurance Dataset.csv")
insurence.columns
insurence.isna().sum()
from sklearn.cluster import KMeans
#generating random uniform numbers
q=np.random.uniform(0,1,)
r=np.random.uniform(0,1,40)
df_qr=pd.DataFrame(columns=['Q','R'])
df_qr.Q=q
df_qr.R=r
df_qr.plot(x='Q',y='R',kind="scatter")
model3=KMeans(n_clusters=5).fit(df_qr)
df_qr.plot(x="Q",y="R",c=model3.labels_,kind="scatter",s=15,cmap=ptl.cm.coolwarm)
