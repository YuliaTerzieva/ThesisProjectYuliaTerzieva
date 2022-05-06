import sys
from math import log
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import rcParams
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


def negLogL(params, frame, GivenData, findLapse, lapsevalue):
    """ This function is a helper one used by scipy.optimize.minimize for a psychometric curve fitting
    :param params: This is array of three values mu, sigma and lapse rate that we want to optimize
    :param frame: single number holding the frame orientation of interest
    :param GivenData: the dataframe with {frame orientations, rod orientations, response - CW or CCW}
    :return: the negative log likelihood of the data given some mu and sigma
    :param findLapse: boolean - if True there is 3rd parameter to be optimized - lapse rate
    :param lapsevalue: numeric value - if findLapse is False, the lapse in the formula is lapsevalue
    """
    mu = params[0]
    sigma = params[1]
    if findLapse:
        lapse = params[2]
    else:
        lapse = lapsevalue
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


def plotFrame(df, frame, lapse=None, findLapse=False, print=False, axs=None):
    # Separating the dataset based on CW and CCW responses on n-1
    CW_index = df.index[df['response'] == 1].tolist()
    CCW_index = df.index[df['response'] == -1].tolist()

    CW_next_index = [i + 1 for i in CW_index if i != df.shape[0] - 1]
    CCW_next_index = [i + 1 for i in CCW_index if i != df.shape[0] - 1]

    CWdata = df.iloc[CW_next_index]
    CCWdata = df.iloc[CCW_next_index]

    CW_post_probs = createMatrixProb(CWdata)
    CCW_post_probs = createMatrixProb(CCWdata)

    if findLapse:
        # For both cases CW and CWW it is important that we put a constraint that sigma and lapse cannot be negative :
        # cons = [{"type": "ineq", "fun": lambda params: params[1]},
        #         {"type": "ineq", "fun": lambda params: params[2]}]
        bnds = ((-30, 30), (0.00001, 20), (0, 0.5))
        CW_results = minimize(negLogL, [0, 2, 0.1], args=(frame, CWdata, findLapse, lapse), bounds=bnds)
        CCW_results = minimize(negLogL, [0, 2, 0.1], args=(frame, CCWdata, findLapse, lapse), bounds=bnds)
    else:
        # For both cases CW and CWW it is important that we put a constraint that sigma and lapse cannot be negative :
        cons = [{"type": "ineq", "fun": lambda params: params[1]}]
        CW_results = minimize(negLogL, [0, 2], args=(frame, CWdata, findLapse, lapse[0]), constraints=cons, )
        CCW_results = minimize(negLogL, [0, 2], args=(frame, CCWdata, findLapse, lapse[1]), constraints=cons)

    if print:
        rod_orients_all = np.sort(df.rodOri.unique())
        frame_orients_all = np.sort(df.frameOri.unique())
        axs.plot(rod_orients_all, CW_post_probs[(np.where(frame_orients_all == frame)[0][0])], "bo")
        axs.plot(rod_orients_all, CCW_post_probs[(np.where(frame_orients_all == frame)[0][0])], "r.")
        axs.plot(rod_orients_all, norm.cdf(rod_orients_all, CW_results.x[0], CW_results.x[1]), "-b",
                 label=f"CW mu {round(CW_results.x[0], 2)}")
        axs.plot(rod_orients_all, norm.cdf(rod_orients_all, CCW_results.x[0], CCW_results.x[1]), "--r",
                 label=f"CCW mu {round(CCW_results.x[0], 2)}")
        axs.legend()
        axs.set_title(f"Frame {frame}")
        axs.label_outer()

    if findLapse:
        # returning mu, sigma and lapse for CW and mu, sigma and lapse for CCW
        return [CW_results.x[0], CW_results.x[1], CW_results.x[2], CCW_results.x[0], CCW_results.x[1],
                CCW_results.x[2]]
    # returning mu and sigma for CW and mu and sigma for CCW
    return [CW_results.x[0], CW_results.x[1], CCW_results.x[0], CCW_results.x[1]]


def obtainLapseRates(nbParticipants, experimentType="frame"):
    """
    This function obtains the lapse rates (individual stimulus independent errors) for CW and CCW previous response
    of all participants for head = 0 and frame = 0
    :param nbParticipants: integer
    :param experimentType: string -> can be "frame", "tilt15", "tilt30"
    :return: double array with size(nbParticipants, 2)
    """
    # There are two lapse rates for each participant - for CW and CCW previous resposne
    lapseRates = np.zeros((nbParticipants, 2))

    for p in range(1, nbParticipants + 1):
        data = pd.read_csv(f'Controls/c{p}/c{p}_{experimentType}.txt', skiprows=13, sep=" ")
        data.drop('reactionTime', inplace=True, axis=1)
        data.drop('Unnamed: 4', inplace=True, axis=1)
        _, _, lapseRates[p - 1][0], _, _, lapseRates[p - 1][1] = plotFrame(data, 0, findLapse=True)

    return lapseRates


def plotAllFramesGivenParticipant(participant, lapseRates, musAndSigmasParticipant, experimentType="frame",
                                  print=False):
    """
    This function loads the data for the given participant, subsequently it flips the negative frames
    of -40, -35, -30, -25, -20, -15, -10, -5 to positive and also flips the rod orientation and
    response for each instance. For each of the resulting 10 frame orientation makes a plot with
    psychometric curves for CW vs CCW previous response
    :param participant: a number between 1 and 16 (including)
    :param lapseRates: tuple - lapse rate for CW and CCW
    :param musAndSigmasParticipant:
    :param experimentType: string -> can be "frame", "tilt15", "tilt30"
    :param print: boolean - when True -> psychometric curves are plotted
    :return: nothing
    """

    # read in the data
    data = pd.read_csv(f'Controls/c{participant}/c{participant}_{experimentType}.txt', skiprows=13, sep=" ")
    # remove the last two columns (these are reactionTime and ??)
    data.drop('reactionTime', inplace=True, axis=1)
    data.drop('Unnamed: 4', inplace=True, axis=1)

    rod_orients_all = np.sort(data.rodOri.unique())
    gravityRod = sum(rod_orients_all) / len(rod_orients_all)
    frame_orients_all = np.sort(data.frameOri.unique())
    frames = [-45, 0, 5, 10, 15, 20, 25, 30, 35, 40]
    for toBeFlipped in list(set(frame_orients_all) - set(frames)):
        indices = data.index[data['frameOri'] == toBeFlipped].tolist()
        for i in indices:
            data.at[i, 'frameOri'] = toBeFlipped * -1
            data.at[i, 'rodOri'] = round(2 * gravityRod - data.at[i, 'rodOri'], 1)
            data.at[i, 'response'] = data.at[i, 'response'] * -1

    if print:
        fig1, axs1 = plt.subplots(5, 2, sharex='all', sharey='all')
        for frame, ax1 in enumerate(axs1.flatten()):
            musAndSigmasParticipant[frame] = plotFrame(data, frames[frame], lapse=lapseRates, print=True, axs=ax1)

        fig1.text(0.5, 0.04, 'Rod orientations', va='center', ha='center', fontsize=rcParams['axes.labelsize'])
        fig1.text(0.04, 0.5, 'P(CW)', va='center', ha='center', rotation='vertical',
                  fontsize=rcParams['axes.labelsize'])
        plt.suptitle(f"Subject {participant}")
        plt.show()
    else:
        for f, frame in enumerate(frames):
            musAndSigmasParticipant[f] = plotFrame(data, frame, lapse=lapseRates)


class InitialAnalysis:
    def __init__(self, nbParticipants, experimentType, plots=False):
        self.nbParticipants = nbParticipants

        lapseRatesParticipants = obtainLapseRates(nbParticipants, experimentType)
        print(f" The lapse rates for participants for experiment {experimentType} are {lapseRatesParticipants}")
        # 16 participants, 10 frames, mu and sigma for CW and mu and sigma for CCW
        self.musAndSigmas = np.zeros((nbParticipants, 10, 4))
        for s in range(1, nbParticipants + 1):
            plotAllFramesGivenParticipant(s, lapseRatesParticipants[s - 1], self.musAndSigmas[s - 1], experimentType,
                                          print=plots)
