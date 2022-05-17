import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.stats.anova import AnovaRM

dmm = pd.read_pickle("dfFrame13fixedSigma")
dmm2 = pd.read_pickle("dfFrame13fixedSigmaSeparateLikelihood")

aov = AnovaRM(dmm,
              depvar='mu',
              subject='nr',
              within=['pre_response', 'frame_orientation']).fit()
print(aov)

# mn = dmm.groupby('frame_orientation').mean()
# plt.plot(mn['mu'])
# # mn['mu'].plot()
# plt.show()
#
# print(mn)

aov2 = AnovaRM(dmm2,
               depvar='mu',
               subject='nr',
               within=['pre_response', 'frame_orientation']).fit()
print(aov2)

# mn2 = dmm2.groupby('frame_orientation').mean()
# plt.plot(mn2['mu'])
# # mn['mu'].plot()
# plt.show()
#
# print(mn2)

# print(dmm['mu'].eq(dmm2['mu']))
print(dmm[abs(dmm['mu'] - dmm2['mu']) > 0.1])
