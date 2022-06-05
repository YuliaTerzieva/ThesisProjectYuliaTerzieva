import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm, binom
from statsmodels.stats.anova import AnovaRM
from initial_data_analysis import createMatrixProb

dataframe0 = pd.read_pickle("NoTiltPlotsAndData/data-Frame-01-06-frame-18-frames")
dataframe15 = pd.read_pickle("Tilt15PlotsAndData/data-Frame-31-05-With-Const-tilt15")
dataframe30 = pd.read_pickle("Tilt30PlotsAndData/data-Frame-01-06-tilt30")
data = [dataframe0, dataframe15, dataframe30]
datanames = ['Head at 0', "Head at 15", "Head at 30"]
plt.subplots(2, 3)
frames = [-45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40]
for d, dataframe in enumerate(data):
    aov = AnovaRM(dataframe,
                  depvar='mu',
                  subject='nr',
                  within=['pre_response', 'frame_orientation']).fit()
    print(aov)

    CWdata = dataframe[dataframe['pre_response'] == 1]
    CCWdata = dataframe[dataframe['pre_response'] == 0]

    CWdataPF = CWdata.groupby('frame_orientation').mean()
    CCWdataPF = CCWdata.groupby('frame_orientation').mean()

    plt.subplot(2, 3, d + 1)
    plt.plot(frames, CWdataPF['mu'], color="#77BAE4")
    plt.plot(frames, CCWdataPF['mu'], color="#E477AD")
    if (d == 0):
        plt.ylabel("Bias ( mu )")
    plt.title(datanames[d])

    plt.subplot(2, 3, d + 4)
    plt.plot(frames, CWdataPF['sigma'], label=f"CW", color="#77BAE4")
    plt.plot(frames, CCWdataPF['sigma'], label=f"CCW", color="#E477AD")
    if (d == 0):
        plt.xlabel("Frame orientation")
    plt.ylabel("Variability ( sigma )")

plt.legend(prop={'size': 8})
plt.tight_layout()
plt.show()

masterDataSet = pd.concat(data)
condition = np.concatenate(([0] * 576, [15] * 576, [30] * 576))
masterDataSet['condition'] = condition
# print(masterDataSet)
# print(masterDataSet.describe())

aov = AnovaRM(masterDataSet,
              depvar='mu',
              subject='nr',
              within=['pre_response', 'frame_orientation', 'condition']).fit()
print(aov)
