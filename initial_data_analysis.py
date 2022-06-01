import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import norm, binom
from scipy.optimize import minimize
from random import randrange


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
            if working_df.shape[0] > 0 and working_df[working_df['response'] == 1].shape[0] > 0:
                probabilities[f][r] = working_df[working_df['response'] == 1].shape[0] / working_df.shape[0]
            else:
                probabilities[f][r] = 0
    return probabilities


def negLogL(params, GivenData):
    """ This function is a helper one used by scipy.optimize.minimize for a psychometric curve fitting
    :param params: This is array of 21 or 37 values depending on the number of frames
                   10/18 mu, 10/18 sigma and 1 lapse rate that we want to optimize
    :param GivenData: the dataframe with {frame orientations, rod orientations, response - CW or CCW}
    :return: the negative log likelihood of the data given the parameters
    """
    rod_orients = np.sort(GivenData.rodOri.unique())
    frame_orients = np.sort(GivenData.frameOri.unique())
    nbFrames = len(frame_orients)
    mus = params[0:nbFrames]
    sigmas = params[nbFrames:nbFrames * 2]
    lapse = params[nbFrames * 2]
    negLogLikelihood = 0

    for f, frame in enumerate(frame_orients):
        curr_df_f = GivenData[(GivenData['frameOri'] == frame)]
        probs = lapse + (1 - 2 * lapse) * norm.cdf(rod_orients, mus[f], sigmas[f])
        for r, rod in enumerate(rod_orients):
            curr_df = curr_df_f[(curr_df_f['rodOri'] == rod)]
            negLogLikelihood += - binom.logpmf(curr_df[curr_df['response'] == 1].shape[0], curr_df['response'].shape[0],
                                               probs[r])

    return negLogLikelihood


def optimizeMSLPlot(df, plot=False):
    """
    This function finds the optimal parameters - mu and sigma for the construction of a psychometric curve
    over the two cases (CW and CCW previous response) for each frame orientation
    :param df: the dataset of a given participant
    :param plot: boolean variable - if True there are plots created and saved
    :return: an array of 2 x 21 where we have 10 mus, 10 sigmas and 1 lapse for CW and CCW case
    """
    # Separating the dataset based on CW and CCW responses on n-1
    CW_index = df.index[df['response'] == 1].tolist()
    CCW_index = df.index[df['response'] == -1].tolist()

    CW_next_index = [i + 1 for i in CW_index if i != df.shape[0] - 1]
    CCW_next_index = [i + 1 for i in CCW_index if i != df.shape[0] - 1]

    CWdata = df.iloc[CW_next_index]
    CCWdata = df.iloc[CCW_next_index]

    nbFrames = len(np.sort(df.frameOri.unique()))
    # mu is between -30 and 30, sigma is between 0.1 and 4 and lapse is between 0 and 0.5
    bnds = np.zeros((2 * nbFrames + 1), dtype=object)
    for i in range(nbFrames):
        # frame = (-10, 10), (0.1, 8), (0, 0.1)
        # tilt15 = (-20, 20), (0.1, 8), (0, 0.1)
        bnds[i] = (-20, 20)
        bnds[i + nbFrames] = (0.1, 8)
    bnds[nbFrames * 2] = (0, 0.1)

    # constrains every next sigma should be bigger or equal than the previous one, not smaller
    const = [{"type": "ineq", "fun": lambda param: param[nbFrames + 1] - param[nbFrames]},
             {"type": "ineq", "fun": lambda param: param[nbFrames + 2] - param[nbFrames + 1]},
             {"type": "ineq", "fun": lambda param: param[nbFrames + 3] - param[nbFrames + 2]},
             {"type": "ineq", "fun": lambda param: param[nbFrames + 4] - param[nbFrames + 3]},
             {"type": "ineq", "fun": lambda param: param[nbFrames + 5] - param[nbFrames + 4]},
             {"type": "ineq", "fun": lambda param: param[nbFrames + 6] - param[nbFrames + 5]},
             {"type": "ineq", "fun": lambda param: param[nbFrames + 7] - param[nbFrames + 6]},
             {"type": "ineq", "fun": lambda param: param[nbFrames + 8] - param[nbFrames + 7]}]

    # the initial parameter guesses are 0 for mu, 1 for sigma and 0.1 for lapse
    parameters = np.zeros((2 * nbFrames + 1))
    parameters[0:nbFrames] = 0
    parameters[nbFrames:nbFrames * 2] = 2
    parameters[nbFrames * 2] = 0.05

    CW_results = minimize(negLogL, parameters, args=CWdata, bounds=bnds)
    print(f"Done with CW minimization the results are {CW_results.x}")
    CCW_results = minimize(negLogL, parameters, args=CCWdata, bounds=bnds)
    print(f"Done with CCW minimization the results are {CCW_results.x}")

    if plot:
        CW_post_probs = createMatrixProb(CWdata)
        CCW_post_probs = createMatrixProb(CCWdata)

        # for condition 'frame' change (6, 3) to (5, 2)
        fig, _ = plt.subplots(6, 3, sharex='all', sharey='all', figsize=(12, 9))
        rod_orients_all = np.sort(df.rodOri.unique())
        frame_orients_all = np.sort(df.frameOri.unique())
        for f, frame in enumerate(frame_orients_all):
            plt.subplot(6, 3, f + 1)
            plt.plot(rod_orients_all, CW_post_probs[(np.where(frame_orients_all == frame)[0][0])], 'o', color="#77BAE4")

            plt.plot(rod_orients_all, CCW_post_probs[(np.where(frame_orients_all == frame)[0][0])], '.', color="#E477AD")

            lapseCW = CW_results.x[nbFrames * 2]
            plt.plot(rod_orients_all,
                     lapseCW + (1 - 2 * lapseCW) * norm.cdf(rod_orients_all, CW_results.x[f], CW_results.x[f + nbFrames]),
                     "#77BAE4",
                     label=f"CW mu {round(CW_results.x[f], 2)}, sigma {round(CW_results.x[f + nbFrames], 2)}")

            lapseCCW = CCW_results.x[nbFrames * 2]
            plt.plot(rod_orients_all,
                     lapseCCW + (1 - 2 * lapseCCW) * norm.cdf(rod_orients_all, CCW_results.x[f], CCW_results.x[f + nbFrames]),
                     "#E477AD",
                     label=f"CCW mu {round(CCW_results.x[f], 2)}, sigma {round(CCW_results.x[f + nbFrames], 2)}")

            plt.legend(prop={'size': 8})
            plt.title(f"Frame {frame}")
            plt.tight_layout()

        fig.add_subplot(111, frame_on=False)
        plt.tick_params(labelcolor="none", bottom=False, left=False)

        plt.xlabel("Rod orientation in degrees")
        plt.ylabel("P(CW)")
        plt.savefig(f'plotTilt30-{randrange(100000)}.png')

    # returning mus and sigmas and lapses for CW and CCW
    return [CW_results.x, CCW_results.x]


def plotAllFramesGivenParticipant(participant, experimentType="frame", plot=False):
    """
    This function loads the data for the given participant,
    subsequently if the experiment type is 'frame' it flips the negative frames
    of -45, -40, -35, -30, -25, -20, -15, -10, -5 to positive and also flips the rod orientation and
    response for each instance. If it is not 'frame' it skips this step.
    For each of the resulting 10/18 frame orientation makes a plot with
    psychometric curves for CW vs CCW previous response
    :param participant: a number between 1 and 16 (including)
    :param experimentType: string -> can be "frame", "tilt15", "tilt30"
    :param plot: boolean - when True -> psychometric curves are plotted and saved
    :return: one vector - mus and sigma for CW and CCW - and a lapse rate for the given participant
    """

    # read in the data
    # [-15. -10.  -5.  -2.   0.   2.   5.  10.  15.]
    data = pd.read_csv(f'Controls/c{participant}/c{participant}_{experimentType}.txt', skiprows=13, sep=" ")
    # remove the last two columns (these are reactionTime and ??)
    data.drop('reactionTime', inplace=True, axis=1)
    data.drop('Unnamed: 4', inplace=True, axis=1)

    if experimentType == "frame":
        rod_orients_all = np.sort(data.rodOri.unique())
        gravityRod = sum(rod_orients_all) / len(rod_orients_all)
        frame_orients_all = np.sort(data.frameOri.unique())
        frames = [0, 5, 10, 15, 20, 25, 30, 35, 40]
        for toBeFlipped in list(set(frame_orients_all) - set(frames)):
            indices = data.index[data['frameOri'] == toBeFlipped].tolist()
            for i in indices:
                data.at[i, 'frameOri'] = toBeFlipped * -1
                data.at[i, 'rodOri'] = round(2 * gravityRod - data.at[i, 'rodOri'], 1)
                data.at[i, 'response'] = data.at[i, 'response'] * -1

    nbFrames = len(np.sort(data.frameOri.unique()))
    resultingMusAndSigmas = optimizeMSLPlot(data, plot=plot)

    musAndSigmasParticipant = [[resultingMusAndSigmas[0][i], resultingMusAndSigmas[1][i],
                                resultingMusAndSigmas[0][i + nbFrames], resultingMusAndSigmas[1][i + nbFrames]]
                               for i in range(nbFrames)]
    # The lapse rates for CW nd CCW
    lapseParticipant = [resultingMusAndSigmas[0][nbFrames * 2], resultingMusAndSigmas[1][nbFrames * 2]]
    return musAndSigmasParticipant, lapseParticipant


class InitialAnalysis:
    def __init__(self, nbParticipants, experimentType, plots=False):

        self.nbParticipants = nbParticipants
        if experimentType == 'frame':
            # 16 participants, 10 frames, mu CW, mu CCW, sigma CW and sigma CCW
            self.musAndSigmas = np.zeros((nbParticipants, 10, 4))
        else:
            # 16 participants, 18 frames, mu CW, mu CCW, sigma CW and sigma CCW
            self.musAndSigmas = np.zeros((nbParticipants, 18, 4))

        self.lapses = np.zeros((nbParticipants, 2))
        for s in range(1, nbParticipants + 1):
            self.musAndSigmas[s - 1], self.lapses[s - 1] = plotAllFramesGivenParticipant(s, experimentType,
                                                                                         plot=plots)
            print(f"Done with participant {s}")
