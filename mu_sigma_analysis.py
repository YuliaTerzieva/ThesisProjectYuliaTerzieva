import math
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_rel
from statsmodels.stats.anova import AnovaRM
from initial_data_analysis import InitialAnalysis
import time


# experiment type can be "frame", "tilt15" or "tilt30"
start = time.time()
analysis = InitialAnalysis(16, experimentType="frame", plots=True)
data = analysis.musAndSigmas
lapses = analysis.lapses
end = time.time()
print(f"Time passed = {end - start}")
print(f"The lapse rates for participants are {lapses}")
# we now have a table which is 16 x 10 x 4 - participants x frames x mus and sigmas
# I'm doing an inference for paired data - t-test
# I want to find the average difference between CW and CCW n-1 train responses for all participants for given frame
pValPerFrame = np.zeros(10)
for f in range(0, 10):
    # the difference for all 16 participants

    diff = data[:, f, 0] - data[:, f, 2]
    print(f" For frame {f} we have difference array {diff}")
    mean_diff = np.mean(diff)
    s_diff = np.std(diff)
    n_diff = len(diff)
    print(f" Mean is {mean_diff} and std is {(s_diff / math.sqrt(n_diff))}")
    # (mean - null hypothesis) / (s/sqrt(n))
    T = (mean_diff - 0) / (s_diff / math.sqrt(n_diff))
    df = n_diff - 1
    print(f" The T values is {T} and df is {df}")
    pValPerFrame[f] = stats.t.sf(np.abs(T), df) * 2

print(pValPerFrame)
pValPerFrame[pValPerFrame > 0.05] = 0
pValPerFrame[pValPerFrame <= 0.05] = 1
print(pValPerFrame)

# second analysis

p = np.zeros(10)
for f in range(0, 10):
    t, p[f] = ttest_rel(data[:, f, 0], data[:, f, 2])

print(p)
frame = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
# third analysis
dm = np.empty((320, 4))
row = 0
for s in range(0, 16):
    for f in range(0, 10):
        for pre in range(0, 2):
            dm[row, :] = [s, frame[f], pre, data[s, f, pre * 2]]
            row = row + 1

dmm = pd.DataFrame(dm, columns=['nr', 'frame_orientation', 'pre_response', 'mu'])

dmm.to_pickle("data-Frame-17-05")

aov = AnovaRM(dmm,
              depvar='mu',
              subject='nr',
              within=['pre_response', 'frame_orientation']).fit()
print(aov)
