# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 08:12:05 2021

@author: chinni
"""


import pandas as pd
lab_tat=pd.read_csv(r"C:\data\data science\Study material\Hypothesis Testing\Datasets_HT\lab_tat_updated.csv")
lab_tat.isna().sum() # cheking na values

#Normality test
import scipy
from scipy import stats
stats.shapiro(lab_tat.Laboratory_1)  # shapiro test
stats.shapiro(lab_tat.Laboratory_2)
stats.shapiro(lab_tat.Laboratory_3)
stats.shapiro(lab_tat.Laboratory_4)  
# p value = 0.32 > 0.05 = Null hypothesis is passed

#Variance test
scipy.stats.levene(lab_tat.Laboratory_1, lab_tat.Laboratory_2,lab_tat.Laboratory_3,lab_tat.Laboratory_4)
# p-value = 0.3810 > 0.05 so p high null fly > Equal variances

# One - Way Anova
F, p = stats.f_oneway(lab_tat.Laboratory_1, lab_tat.Laboratory_2,lab_tat.Laboratory_3,lab_tat.Laboratory_4)
print('p value:',p)

#p-value < 0.05
# All the 3 laboratories have equal mean difference in TAT time
