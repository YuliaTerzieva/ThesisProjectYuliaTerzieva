import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.stats.anova import AnovaRM

dmm = pd.read_pickle("data-Frame-23-05-no-const")

# aov = AnovaRM(dmm,
#               depvar='mu',
#               subject='nr',
#               within=['pre_response', 'frame_orientation']).fit()
# print(aov)

# print(dmm.head())
# print(dmm.shape)
# new = dmm.groupby('frame_orientation', 'pre_response').mean()
# print(new)

musDifference = np.zeros(10)
frames = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
for f, frame in enumerate(frames):
    cur = dmm[dmm['frame_orientation'] == frame]
    curr = cur.groupby('pre_response').mean()
    musDifference[f] = curr['mu'][1]-curr['mu'][0]

plt.plot(frames, musDifference)
plt.xlabel("Frame orientation")
plt.ylabel("Difference between CW and CCW ")
plt.show()

mn = dmm.groupby('frame_orientation').mean()
print(mn)
plt.plot(mn['mu'])
plt.show()

print(mn)

