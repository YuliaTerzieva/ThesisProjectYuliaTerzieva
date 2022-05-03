import math
import numpy as np
from scipy import stats

from initial_data_analysis import InitialAnalysis

analysis = InitialAnalysis(16)
data = analysis.musAndSigmas

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
    pValPerFrame[f] = stats.t.sf(np.abs(T), df) #* 2

print(pValPerFrame)
pValPerFrame[pValPerFrame > 0.05] = 1
pValPerFrame[pValPerFrame <= 0.05] = 0
print(pValPerFrame)
