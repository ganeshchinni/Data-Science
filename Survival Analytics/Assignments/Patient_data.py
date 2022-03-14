# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 22:57:03 2022

@author: chinni
"""

import pandas as pd
new_patient_data=pd.read_csv(r"C:\data\data science\Study material\Survival Analytics\Datasets_Survival Analytics\Patient.csv")
#new_patient_data=patient_data.drop('PatientID',axis=1)
new_patient_data.Scenario=1
# Followup is referring to time 
T = new_patient_data.Followup

# Importing the KaplanMeierFitter model to fit the survival analysis
pip install lifelines
from lifelines import KaplanMeierFitter
#initilizing the KaplanMeierFitter
kmf=KaplanMeierFitter()

# Fitting KaplanMeierFitter model on Time and Events for death 
kmf.fit(T, event_observed=new_patient_data.Eventtype)

# Time-line estimations plot 
kmf.plot()

# Over Multiple groups 
# For each group, here group is Scenario
new_patient_data.Scenario.value_counts()

# Applying KaplanMeierFitter model on Time and Events for the group "1"
kmf.fit(T[new_patient_data.Scenario==1], new_patient_data.Eventtype[new_patient_data.Scenario==1], label='1')
ax = kmf.plot()

