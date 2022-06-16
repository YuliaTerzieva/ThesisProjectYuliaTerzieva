import numpy as np
import pandas as pd
from initial_data_analysis import InitialAnalysis

# experiment type can be "frame", "tilt15" or "tilt30"
headOrientation = "frame"
analysis = InitialAnalysis(16, experimentType=headOrientation, plots=True)
data = analysis.musAndSigmas
lapses = analysis.lapses
print(f"The lapse rates for participants are {lapses}")

frame = [-45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40]
matrix = np.empty((576, 5))
row = 0
for s in range(0, 16):
    for f in range(0, 18):
        for pre in range(0, 2):
            # pre = 0 is now CW and pre 1 is CCW
            matrix[row, :] = [s, frame[f], pre, data[s, f, pre], data[s, f, (pre + 2)]]
            row = row + 1

dataframe = pd.DataFrame(matrix, columns=['nr', 'frame_orientation', 'pre_response', 'mu', 'sigma'])

dataframe.to_pickle(f"DataFrame-{headOrientation}")


