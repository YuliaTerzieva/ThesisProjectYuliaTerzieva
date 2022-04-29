import sys
from math import log
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import rcParams, style
from scipy.stats import norm, binom
from scipy.optimize import minimize


# Todo: 1) There should be constrain as lapse rate cannot go under 0
#       done -> 2) Problems with likelihood becoming 0 and log of 0 is undefined
#       done -> 3) In plotSingleSubject remove param rod_orientations
#       as this can be obtained in the function negLogL from GivenData
#       4) Change the plotting to seaborn and not matplotlib

def createMatrixProb(dataframe):
    """
    This function creates a matrix [frame orientations, rod orientations],
    where each position has the probability of answering CW for this combination of frame and rod
    (* From the data how many times did the participant answer CW out of the total amount of times
    he saw this frame/rod combination*)
    :param dataframe: with columns frame_orientation, rod_orientation, response
    :return: probabilities - the matrix
    """
    # check the rod and frame orientations (rods depend on subject and condition)
    rod_orients = np.sort(dataframe.rodOri.unique())
    frame_orients = np.sort(dataframe.frameOri.unique())

    # A matrix with rows the frame orientations and columns rod orientations
    # where each frame/rod combination is the number of CW responses out of 10
    probabilities = np.zeros((len(frame_orients), len(rod_orients)))

    for f, frame in enumerate(frame_orients):
        for r, rod in enumerate(rod_orients):
            # working_df are the rows in the dataset where we have the given frame and rod
            working_df = dataframe[(dataframe['frameOri'] == frame) & (dataframe['rodOri'] == rod)]
            if working_df.shape[0] > 0:
                probabilities[f][r] = working_df[working_df['response'] == 1].shape[0] / working_df.shape[0]
            else:
                probabilities[f][r] = 0

    return probabilities


def negLogL(params, frame, GivenData):
    """ This function is a helper one used by scipy.optimize.minimize for a psychometric curve fitting
    :param params: This is array of three values mu, sigma and lapse rate that we want to optimize
    :param frame: single number holding the frame orientation of interest
    :param GivenData: the dataframe with {frame orientations, rod orientations, response - CW or CCW}
    :return: the negative log likelihood of the data given some mu and sigma
    """
    mu = params[0]
    sigma = params[1]
    lapse = params[2]
    rod_orients = np.sort(GivenData.rodOri.unique())
    negLogLikelihood = 0

    for rod in rod_orients:
        curr_df = GivenData[(GivenData['frameOri'] == frame) & (GivenData['rodOri'] == rod)]
        probs = lapse + (1 - 2 * lapse) * norm.cdf(rod, mu, sigma)
        response = curr_df['response']
        response = response.tolist()
        likelihood = binom.pmf(response.count(1), len(response), probs)
        if likelihood == 0:
            likelihood = sys.float_info.min
        negLogLikelihood += - log(likelihood)
    return negLogLikelihood


def plotAllSubjectsOneFrame(subject, axs, frame):
    # read in the data
    df = pd.read_csv(f'Controls/c{subject}/c{subject}_frame.txt', skiprows=13, sep=" ")
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

    # For both cases CW and CWW it is important that we put a constraint that sigma and lapse cannot be negative :
    cons = [{"type": "ineq", "fun": lambda params: params[1]},
            {"type": "ineq", "fun": lambda params: params[2]}]

    # Obtaining statistics for n-1 = CW frame = 0
    CW_post_probs = createMatrixProb(CWdata)
    print("***** START OF CW DATA ANALYSIS *****")
    CW_results = minimize(negLogL, [0, 2, 0.1], args=(frame, CWdata), constraints=cons, )
    print(
        f" Done with CW results {subject}: mu = {CW_results.x[0]}, sigma = {CW_results.x[1]}, lapse = {CW_results.x[2]}")

    # Obtaining statistics for n-1 = CCW frame = 0
    CCW_post_probs = createMatrixProb(CCWdata)
    print("***** START OF CCW DATA ANALYSIS *****")
    CCW_results = minimize(negLogL, [0, 2, 0.1], args=(frame, CCWdata), constraints=cons)
    print(
        f" Done with CCW results {subject}: mu = {CCW_results.x[0]}, sigma = {CCW_results.x[1]}, lapse = {CCW_results.x[2]}")

    # plot the corresponding psychometric curves
    axs.plot(rod_orients_all, CW_post_probs[(np.where(frame_orients_all == frame)[0][0])], "bo")
    axs.plot(rod_orients_all, CCW_post_probs[(np.where(frame_orients_all == frame)[0][0])], "r.")
    axs.plot(rod_orients_all, norm.cdf(rod_orients_all, CW_results.x[0], CW_results.x[1]), "-b",
             label=f"CW mu {round(CW_results.x[0], 2)}")
    axs.plot(rod_orients_all, norm.cdf(rod_orients_all, CCW_results.x[0], CCW_results.x[1]), "--r",
             label=f"CCW mu {round(CCW_results.x[0], 2)}")
    axs.legend()
    axs.set_title(f"Subject {subject}")
    axs.label_outer()


fig1, axs1 = plt.subplots(4, 2, sharex='all', sharey='all')
fig2, axs2 = plt.subplots(4, 2, sharex='all', sharey='all')

for subject, (ax1, ax2) in enumerate(zip(axs1.flatten(), axs2.flatten())):
    plotAllSubjectsOneFrame(subject + 1, ax1, 0)
    plotAllSubjectsOneFrame(subject + 9, ax2, 0)

fig1.text(0.5, 0.04, 'Rod orientations', va='center', ha='center', fontsize=rcParams['axes.labelsize'])
fig1.text(0.04, 0.5, 'P(CW)', va='center', ha='center', rotation='vertical', fontsize=rcParams['axes.labelsize'])
fig2.text(0.5, 0.04, 'Rod orientations', va='center', ha='center', fontsize=rcParams['axes.labelsize'])
fig2.text(0.04, 0.5, 'P(CW)', va='center', ha='center', rotation='vertical', fontsize=rcParams['axes.labelsize'])
plt.show()