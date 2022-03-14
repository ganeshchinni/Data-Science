# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 09:10:39 2021

@author: chinni
"""


import pandas as pd
import scipy
from scipy import stats

Fantaloons = pd.read_csv(r"C:\data\data science\Study material\Hypothesis Testing\Datasets_HT\Fantaloons.csv")

count = pd.crosstab(Fantaloons["Weekdays"], Fantaloons["Weekend"])
count

Chisquares_results = scipy.stats.chi2_contingency(count)

Chi_square = [['Test Statistic', 'p-value'], [Chisquares_results[0], Chisquares_results[1]]]
Chi_square
