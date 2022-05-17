from scipy.stats import norm, binom
import pandas as pd
import numpy as np

dmm = pd.read_pickle("dfFrame13fixedSigma")
dmm2 = pd.read_pickle("dfFrame13fixedSigmaSeparateLikelihood")
print(dmm.head())
print(dmm2.head())
difference = abs(dmm['mu'] - dmm2['mu'])

difference[difference > 0.05] = 1
difference[difference <= 0.05] = 0

print("we have ", difference.value_counts())