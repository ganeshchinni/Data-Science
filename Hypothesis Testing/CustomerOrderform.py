# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 09:02:41 2021

@author: chinni
"""

import pandas as pd
import scipy
from scipy import stats

customers = pd.read_csv(r"C:\data\data science\Study material\Hypothesis Testing\Datasets_HT\CustomerOrderform.csv")

count = pd.crosstab(customers["Phillippines"], customers["Indonesia"])
count

Chisquares_results = scipy.stats.chi2_contingency(count)

Chi_square = [['Test Statistic', 'p-value'], [Chisquares_results[0], Chisquares_results[1]]]
Chi_square

count1 = pd.crosstab(customers["Indonesia"], customers["Malta"])
count1

Chisquares_results = scipy.stats.chi2_contingency(count1)

Chi_square = [['Test Statistic', 'p-value'], [Chisquares_results[0], Chisquares_results[1]]]
Chi_square

count2 = pd.crosstab(customers["Malta"], customers["India"])
count2

Chisquares_results = scipy.stats.chi2_contingency(count2)

Chi_square = [['Test Statistic', 'p-value'], [Chisquares_results[0], Chisquares_results[1]]]
Chi_square

