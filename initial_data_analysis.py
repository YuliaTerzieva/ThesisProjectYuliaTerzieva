from math import log
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize


def negLogL(params, rod_orients, frame, GivenData, comments):
    """ This function is a helper one used by scipy.optimize.minimize for a psychometric curve fitting
    :param params: This is array of two values mu and sigma that we want to optimize
    :param rod_orients: array of possible rod orientations TODO:this param can be removed and obtained through GivenData
    :param frame: single number holding the frame orientation of interest
    :param GivenData: the dataframe with {frame orientations, rod orientations, response - CW or CCW}
    :param comments: boolean - True is used for debugging TODO: remove this when done
    :return: the negative log likelihood of the data given some mu and sigma
    """
    mu = params[0]
    sigma = params[1]
    if comments:
        print(f" mu = {mu} and sigma = {sigma}")
    negLogLikelihood = 0
    for rod in rod_orients:
        curr_df = GivenData[(GivenData['frameOri'] == frame) & (GivenData['rodOri'] == rod)]
        probs = norm.cdf(rod, mu, sigma)
        if comments:
            print(f"for rod {rod} we have probability of CW = {probs}")
        response = curr_df['response']
        response = response.tolist()
        if comments:
            print(f" the responses for this rod are {response}")
            print(f" the CW responses are {response.count(1)} and CCW {response.count(-1)}")
        likelihood = response.count(1) * probs + response.count(-1) * (1 - probs)
        if comments:
            print(f" The likelihood is {likelihood}")
            print(f" The log likelihood is {log(likelihood)}")
        negLogLikelihood += - log(likelihood)
    return negLogLikelihood


def createMatrixProb(dataframe):
    # check the rod and frame orientations (rods depend on subject and condition)
    rod_orients = np.sort(dataframe.rodOri.unique())
    frame_orients = np.sort(dataframe.frameOri.unique())

    # A matrix with rows the frame orientations and columns rod orientations
    # where each frame/rod combination is the number of CW responses out of 10
    probabilities = np.zeros((len(frame_orients), len(rod_orients)))

    for f, frame in enumerate(frame_orients):
        for r, rod in enumerate(rod_orients):
            # Todone : 1) Now i need to make it such that the points are not out ot 10, but out of depending on the rod,
            # because when i separate the data the rods won't be out of 10. -> I achieved this by first selecting only the
            # data with frame and rod and the diving the CW responses by all the responses

            # working_df are the rows in the dataset where we have the given frame and rod
            working_df = dataframe[(dataframe['frameOri'] == frame) & (dataframe['rodOri'] == rod)]
            if working_df.shape[0] > 0:
                probabilities[f][r] = working_df[working_df['response'] == 1].shape[0] / working_df.shape[0]
            else:
                probabilities[f][r] = 0

    return probabilities


# THIS WORKS WELL FOR C2 AND C6 haven't check after that.
# read in the data
df = pd.read_csv('Controls/c2/c2_frame.txt', skiprows=13, sep=" ")
# remove the last two columns (these are reactionTime and ??)
df.drop('reactionTime', inplace=True, axis=1)
df.drop('Unnamed: 4', inplace=True, axis=1)
rod_orients_all = np.sort(df.rodOri.unique())
frame_orients_all = np.sort(df.frameOri.unique())

# Separating the dataset based on CW and CCW responses on n-1
CW_index = df.index[df['response'] == 1].tolist()
CCW_index = df.index[df['response'] == -1].tolist()

CW_next_index = [i + 1 for i in CW_index if i != df.shape[0] - 1]
CCW_next_index = [i + 1 for i in CCW_index if i != df.shape[0] - 1]

CWdata = df.iloc[CW_next_index]
CCWdata = df.iloc[CCW_next_index]

# For both cases CW and CWW it is important that we put a constraint that sigma cannot be negative :
cons = ({"type": "ineq", "fun": lambda params: params[1]})

# Obtaining statistics for n-1 = CW frame = 0
# For the minimize in general I hough I should pass as a parameter for the rods np.(CWdata.rodOri.unique())
# because it is possible that after separating the dataset there are some rod representations missing, but turns out
# there is no such problem so for simplicity I used rod_orients_all . PS i can also remove this parameter and obrain the
# rod orients in the function itself using givendata/CWdata/CCWdta
CW_post_probs = createMatrixProb(CWdata)
print("***** START OF CW DATA ANALYSIS *****")
CW_results = minimize(negLogL, [1, 1.5], args=(rod_orients_all, 0, CWdata, False), constraints=cons)
print(f" Done with CW results : mu = {CW_results.x[0]} and sigma = {CW_results.x[1]}")

# Obtaining statistics for n-1 = CCW frame = 0
CCW_post_probs = createMatrixProb(CCWdata)
print("***** START OF CCW DATA ANALYSIS *****")
CCW_results = minimize(negLogL, [1, 1.5], args=(rod_orients_all, 0, CCWdata, False), constraints=cons)
print(f" Done with CCW results : mu = {CCW_results.x[0]} and sigma = {CCW_results.x[1]}")

plt.plot(np.sort(CWdata.rodOri.unique()), CW_post_probs[(np.where(frame_orients_all == 0)[0][0])], "bo",
         label="CW data")
plt.plot(np.sort(CCWdata.rodOri.unique()), CCW_post_probs[(np.where(frame_orients_all == 0)[0][0])], "r.",
         label="CCW data")
plt.plot(rod_orients_all, norm.cdf(rod_orients_all, CW_results.x[0], CW_results.x[1]), "-k", label="curve CW")
plt.plot(rod_orients_all, norm.cdf(rod_orients_all, CCW_results.x[0], CCW_results.x[1]), "--c", label="curve CCW")
plt.xlabel("Rod orientations in degrees")
plt.ylabel("P(CW)")
plt.title("For head = 0 and frame = 0")
plt.legend()
plt.show()
