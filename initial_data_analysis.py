import math

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

def negLogL(params, rod_orients, frame, df):
    """ This function is a helper one used by scipy.optimize.minimize
    in order for a psychometric curve fitting
    :param params: This is array of two values mu and sigma that we want to optimize
    :param rod_orients: array of possible rod orientations
    :param frame: single number holding the frame orientation of interest
    :return: the negative log likelihood of the data given some mu and sigma
    """
    mu = params[0]
    sigma = params[1]
    negLogLikelihood = 0
    for rod in rod_orients:
        curr_df = df[(df['frameOri'] == frame) & (df['rodOri'] == rod)]
        probs = norm.cdf(rod, mu, sigma)
        # print(f"for rod {rod} we have probability of CW = {probs}")
        response = curr_df['response']
        response = response.tolist()
        # print(f" the responses for this rod are {response}")
        # print(f" the CW responses are {response.count(1)} and CCW {response.count(-1)}")
        likelihood = response.count(1) * probs + response.count(-1) * (1 - probs)
        # print(f" The likelihood is {likelihood}")
        # print(f" The log likelihood is {math.log(likelihood)}")
        negLogLikelihood += - math.log(likelihood)
    return negLogLikelihood


# read in the data
df = pd.read_csv('Controls/c2/c2_frame.txt', skiprows=13, sep=" ")
# remove the last two columns (these are reactionTime and ??)
df.drop('reactionTime', inplace=True, axis=1)
df.drop('Unnamed: 4', inplace=True, axis=1)
# check the rod and frame orientations (rods depend on subject and condition)
rod_orients = np.sort(df.rodOri.unique())
frame_orients = np.sort(df.frameOri.unique())

# A matrix with rows the frame orientations and columns rod orientations
# where each frame/rod combination is the number of CW responses out of 10
probabilities = np.zeros((len(frame_orients), len(rod_orients)))

for f, frame in enumerate(frame_orients):
    for r, rod in enumerate(rod_orients):
        # Todone : 1) Now i need to make it such that the points are not out ot 10, but out of depending on the rod,
        # because when i separate the data the rods won't be out of 10. -> I achieved this by first selecting only the
        # data with frame and rod and the diving the CW responses by all the responses

        # working_df are the rows in the dataset where we have the given frame and rod
        working_df = df[(df['frameOri'] == frame) & (df['rodOri'] == rod)]
        probabilities[f][r] = working_df[working_df['response'] == 1].shape[0] / working_df.shape[0]

# Test for frame 0 degrees
results = minimize(negLogL, [0, 2], args=(rod_orients, 0, df))
print(results)
print(f" The mu is {results.x[0]} and sigma is {results.x[0]}")

plt.plot(rod_orients, probabilities[(np.where(frame_orients == 0)[0][0])], "o")
plt.plot(rod_orients, norm.cdf(rod_orients, results.x[0], results.x[1]))
plt.xlabel("Rod orientations in degrees")
plt.ylabel("P(CW)")
plt.title("For head = 0 and frame = 0")
plt.show()







