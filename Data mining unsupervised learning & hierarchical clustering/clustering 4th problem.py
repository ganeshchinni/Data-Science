# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 16:59:52 2021

@author: chinni
"""

'''4.Perform clustering on mixed data. Convert the categorical variables to numeric by using
    dummies or label encoding and perform normalization techniques. The data set consists of
    details of customers related to their auto insurance. Refer to Autoinsurance.csv dataset.'''
    
    
import pandas as pd
insurance=pd.read_csv(r"C:\data\data science\Study material\Data mining unsupervised learning & hierarchical clustering\Dataset_Assignment Clustering\AutoInsurance.csv")
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

#standardization 
from sklearn.preprocessing import StandardScaler
#initialization of scalar
scalar=StandardScaler()
#scaling data
insurance_std=scalar.fit_transform(insurance)
#converting array back to data frame
dataset=pd.DataFrame(insurance_std)
res_std=dataset.describe()

#for creating dendogram
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
z=linkage(insurance_norm, method = "complete", metric = "euclidean")
# dendogram
import matplotlib.pyplot as plt
plt.figure(figsize=(30, 20));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 45,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()

