# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 07:54:24 2021

@author: chinni
"""

import pandas as pd
cutlets=pd.read_csv(r"C:\data\data science\Study material\Hypothesis Testing\Datasets_HT\Cutlets.csv")
cutlets.isna().sum() # cheking na values
cutlets.dropna(axis=0, inplace=True) # dropping null cells
cutlets.columns='Unit_A','Unit_B' # renaming column names

#Normality test
import scipy
from scipy import stats
stats.shapiro(cutlets.Unit_A)  # shapiro test
stats.shapiro(cutlets.Unit_B)

#Variance test
scipy.stats.levene(cutlets.Unit_A,cutlets.Unit_B)
# p-value = 0.4176 > 0.05 so p high null fly > Equal variances

# 2 Sample T test
scipy.stats.ttest_ind(cutlets.Unit_A,cutlets.Unit_B)
#p-value > 0.05
#p-value=0.4722394724599501 there is no difference in diameter because of high p values


