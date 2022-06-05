from scipy.stats import norm, binom
import pandas as pd
import numpy as np

# dmm = pd.read_pickle("dfFrame13fixedSigma")
# dmm2 = pd.read_pickle("dfFrame13fixedSigmaSeparateLikelihood")
# print(dmm.head())
# print(dmm2.head())
# difference = abs(dmm['mu'] - dmm2['mu'])
#
# difference[difference > 0.05] = 1
# difference[difference <= 0.05] = 0
#
# print("we have ", difference.value_counts())
#
# test = np.linspace(0, 9, 10)
# print(test)
# print(test[2:])
#
# print(np.sum(test))
# print(np.sum(test[2:]))

# head_space_orients = np.linspace(-180, 180, 361)
#
# print(np.linspace(-45, 40, 18))
# # print(head_space_orients)
# print(np.where(head_space_orients == 0))


data = pd.read_csv(f'Controls/c{10}/c{10}_tilt30.txt', skiprows=13, sep=" ")
# remove the last two columns (these are reactionTime and ??)
data.drop('reactionTime', inplace=True, axis=1)
data.drop('Unnamed: 4', inplace=True, axis=1)

rod_orients_all = np.sort(data.rodOri.unique())
print(rod_orients_all)