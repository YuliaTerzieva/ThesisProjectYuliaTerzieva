import numpy as np
import pandas as pd
from statsmodels.stats.anova import AnovaRM
from initial_data_analysis import InitialAnalysis
import time

# experiment type can be "frame", "tilt15" or "tilt30"
start = time.time()
analysis = InitialAnalysis(16, experimentType="frame", plots=True)
data = analysis.musAndSigmas
print(data)
lapses = analysis.lapses
end = time.time()
print(f"Time passed = {end - start}")
print(f"The lapse rates for participants are {lapses}")

# frame = [-45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40] # <- for tilts
frame = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45] # <- for frame
# matrix = np.empty((576, 5)) # <- for tilts
matrix = np.empty((320, 5)) # <- for frame
row = 0
for s in range(0, 16):
    for f in range(0, 10): # <- for frame is 10 for tilts is 18
        for pre in range(0, 2):
            matrix[row, :] = [s, frame[f], pre, data[s, f, pre], data[s, f, (pre + 2)]]
            row = row + 1

dataframe = pd.DataFrame(matrix, columns=['nr', 'frame_orientation', 'pre_response', 'mu', 'sigma'])

dataframe.to_pickle("data-Frame-01-06-frame-10-frames")

aov = AnovaRM(dataframe,
              depvar='mu',
              subject='nr',
              within=['pre_response', 'frame_orientation']).fit()
print(aov)

