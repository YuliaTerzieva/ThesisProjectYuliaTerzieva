import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.stats.anova import AnovaRM


dmm = pd.read_pickle("dataframemus")
dmm.rename(columns={'pre-response':'pre_response'}, inplace=True)
print(dmm.head())
aov = AnovaRM(dmm,
              depvar='mu',
              subject='nr',
              within=['pre_response', 'frame_orientation']).fit()
print(aov)

mn = dmm.groupby('frame_orientation').mean()
plt.plot(mn['mu'])
# mn['mu'].plot()
plt.show()

print(mn)