# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 08:24:17 2021

@author: chinni
"""


import pandas as pd
from scipy import stats
BuyerRatio=pd.read_csv(r"C:\data\data science\Study material\Hypothesis Testing\Datasets_HT\BuyerRatio.csv")
BuyerRatio.rename(columns={"Observed Values":"Gender"}, inplace=True)
BuyerRatio.columns
BuyerRatio["Gender"].replace("Males", 0 , inplace = True)
BuyerRatio["Gender"].replace("Females", 1 , inplace = True)
BuyerRatio

Chisquare_results = stats.chi2_contingency(BuyerRatio)
Chisquare_results 

Chi_square = [["Test statistics", "p-value"],[Chisquare_results[0],Chisquare_results[1]]]
Chi_square
#AS p = 0.7919942975413565 p > 0.05 so Fails to reject null hypothesis
count = pd.crosstab(BuyerRatio["East"], BuyerRatio["West"])
print(count)

Chisquares_results = stats.chi2_contingency(count)
print(Chisquares_results)
Chi_square = [['Test Statistic', 'p-value'], [Chisquares_results[0], Chisquares_results[1]]]
Chi_square
