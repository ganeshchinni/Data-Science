# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 10:37:41 2022

@author: chinni
"""

import pandas as pd
import numpy as np
ECG_data=pd.read_excel(r"C:\data\data science\Study material\Survival Analytics\Datasets_Survival Analytics\ECG_Surv.xlsx")
ECG_data.isna().sum()
from sklearn.impute import SimpleImputer
mean_imputer=SimpleImputer(missing_values=np.nan,strategy="mean")
ECG_data['epss']=pd.DataFrame(mean_imputer.fit_transform(ECG_data[['epss']]))
ECG_data['lvdd']=pd.DataFrame(mean_imputer.fit_transform(ECG_data[['lvdd']]))
ECG_data['wallmotion-score']=pd.DataFrame(mean_imputer.fit_transform(ECG_data[['wallmotion-score']]))
ECG_data['wallmotion-index']=pd.DataFrame(mean_imputer.fit_transform(ECG_data[['wallmotion-index']]))
ECG_data['multi_sensor']=pd.DataFrame(mean_imputer.fit_transform(ECG_data[['multi_sensor']]))
#removing name column
ECG_data=ECG_data.drop('name',axis=1)

# Spell is referring to time 
T = ECG_data.survival_time_hr

# Importing the KaplanMeierFitter model to fit the survival analysis
from lifelines import KaplanMeierFitter

# Initiating the KaplanMeierFitter model
kmf = KaplanMeierFitter()

# Fitting KaplanMeierFitter model on Time and Events for death 
kmf.fit(T, event_observed=ECG_data.alive)

# Time-line estimations plot 
kmf.plot()

# Over Multiple groups 
# For each group, here group is ui
ECG_data.group.value_counts()

# Applying KaplanMeierFitter model on different 1st group of the data
kmf.fit(T[ECG_data.group==1], ECG_data.alive[ECG_data.group==1], label='1')
ax = kmf.plot()

# Applying KaplanMeierFitter model on different 1st group of the data
kmf.fit(T[ECG_data.group==2], ECG_data.alive[ECG_data.group==2], label='2')
ay = kmf.plot()

# Applying KaplanMeierFitter model on different 1st group of the data
kmf.fit(T[ECG_data.group==3], ECG_data.alive[ECG_data.group==3], label='3')
az = kmf.plot()
