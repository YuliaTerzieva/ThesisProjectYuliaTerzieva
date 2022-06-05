import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import binom
from scipy.optimize import minimize
from scipy.interpolate import make_interp_spline
from VisualVestibularVerticalityModel import VisualVestibularVerticalityModel
from statsmodels.stats.anova import AnovaRM
from initial_data_analysis import createMatrixProb


def negLogL(params, GivenData):
    """ This function is a helper one used by scipy.optimize.minimize for a parameter estimation
    :param params: [sigma_hp, alpha_hs, beta_hs, k_ver, k_hor, tau, lapse]
    :param GivenData: the dataframe with {frame orientations, rod orientations, response - CW or CCW}
    :return: the negative log likelihood of the data given the parameters
    """

    rod_orients = np.sort(GivenData.rodOri.unique())
    frame_orients = np.sort(GivenData.frameOri.unique())
    lapse = params[6]

    negLogLikelihood = 0
    for f, frame in enumerate(frame_orients):
        # the parameters - the head is set to 0 the rest is from the parameters we are optimizing
        model = VisualVestibularVerticalityModel(np.append([0, frame], params[:6]))
        curr_df_f = GivenData[(GivenData['frameOri'] == frame)]
        for r, rod in enumerate(rod_orients):
            probs = lapse + (1 - 2 * lapse) * model.getCWProbability(rod)
            curr_df = curr_df_f[(curr_df_f['rodOri'] == rod)]
            negLogLikelihood += - binom.logpmf(curr_df[curr_df['response'] == 1].shape[0], curr_df['response'].shape[0],
                                               probs)
    return negLogLikelihood


def modelFitting():
    participant = 7
    experimentType = 'frame'
    data = pd.read_csv(f'Controls/c{participant}/c{participant}_{experimentType}.txt', skiprows=13, sep=" ")
    data.drop('reactionTime', inplace=True, axis=1)
    data.drop('Unnamed: 4', inplace=True, axis=1)

    # params = {sigma_hp, alpha_hs, beta_hs, k_ver, k_hor, tau, lapse}
    parameters = np.array([6.5, 0.07, 2.21, 4.87, 52.26, 0.4, 0.05])
    bnds = [(0, 9), (0, 1), (0, 4), (0, 8), (0, 100), (0, 1), (0, 0.1)]

    result = minimize(negLogL, parameters, args=data, bounds=bnds)

    print(result)
    print(result.x)
    print(type(result.x))


def plotsFigurePost():
    model00 = VisualVestibularVerticalityModel([0, 0, 6.5, 0.07, 2.21, 4.87, 52.26, 0.8, 0.1])
    model300 = VisualVestibularVerticalityModel([30, 0, 6.5, 0.07, 2.21, 4.87, 52.26, 0.8, 0.1])
    model020 = VisualVestibularVerticalityModel([0, 20, 6.5, 0.07, 2.21, 4.87, 52.26, 0.8, 0.1])
    model00.head_in_space_posterior()
    model300.head_in_space_posterior()
    model020.head_in_space_posterior()
    plt.subplots(2, 2)
    head_space_orients = model00.head_space_orients

    plt.subplot(2, 2, 1)
    plt.plot(head_space_orients, model00.head_space_prior, '#B081D9')
    plt.xlim(- 20, 20)
    plt.xlabel("Degrees")
    plt.title("Head-in-space prior")

    plt.subplot(2, 2, 2)
    plt.plot(head_space_orients, model00.vest_likelihood, '#B081D9', label=f'Head {model00.H_true}')
    plt.plot(head_space_orients, model300.vest_likelihood, '#77BAE4', label=f'Head {model300.H_true}')
    plt.plot(head_space_orients, model020.vest_likelihood, '#E477AD', label=f'Head {model020.H_true}')
    plt.xlim(- 20, 50)
    plt.xlabel("Degrees")
    plt.title(f"Vestibular likelihood")

    plt.subplot(2, 2, 3)
    plt.plot(head_space_orients, model00.cont_likelihood, '#B081D9', label=f'Frame {model00.F_true}')
    plt.plot(head_space_orients, model300.cont_likelihood, '#77BAE4', label=f'Frame {model300.F_true}')
    plt.plot(head_space_orients, model020.cont_likelihood, '#E477AD', label=f'Frame {model020.F_true}')
    plt.xlabel("Degrees")
    plt.title(f"Contextual likelihood")

    plt.subplot(2, 2, 4)
    plt.plot(head_space_orients, model00.head_space_post, '#B081D9',
             label=f'Head {model00.H_true}, Frame {model00.F_true}')
    plt.plot(head_space_orients, model300.head_space_post, '#77BAE4',
             label=f'Head {model300.H_true}, Frame {model300.F_true}')
    plt.plot(head_space_orients, model020.head_space_post, '#E477AD',
             label=f'Head {model020.H_true}, Frame {model020.F_true}')
    plt.xlabel("Degrees")
    plt.title("Head-in-space posterior")
    plt.xlim(- 20, 50)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plotsFigureCW():
    model00 = VisualVestibularVerticalityModel([0, 0, 6.5, 0.07, 2.21, 4.87, 52.26, 0.8, 0.1])
    model300 = VisualVestibularVerticalityModel([30, 0, 6.5, 0.07, 2.21, 4.87, 52.26, 0.8, 0.1])
    model020 = VisualVestibularVerticalityModel([0, 20, 6.5, 0.07, 2.21, 4.87, 52.26, 0.8, 0.1])

    rods = np.linspace(-15, 15, 101)
    plt.plot(rods, model00.getCWProbability(rods), '#B081D9', label="Head 0, Frame 0")
    plt.plot(rods, model300.getCWProbability(rods), '#77BAE4', label="Head 30, Frame 0")
    plt.plot(rods, model020.getCWProbability(rods), '#E477AD', label="Head 0, Frame 20")
    plt.legend()
    plt.xlabel("Rod orientation")
    plt.ylabel("Probability of CW response")
    plt.show()


def plotsFigureCWPSE():
    frames = [-45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40]
    biases = np.zeros(18)

    for f, frame in enumerate(frames):
        model_curr = VisualVestibularVerticalityModel([0, frame, 6.5, 0.07, 2.21, 4.87, 52.26, 0.8, 0.1])
        rods = np.linspace(-15, 15, 201)
        curve = model_curr.getCWProbability(rods)
        # plt.plot(rods, curve, label=f"Frame {frame}")
        biases[f] = rods[(np.abs(curve - 0.5)).argmin()]

    # plt.xlim((-6, 10))
    # plt.legend()
    # plt.show()
    frames_new = np.linspace(-45, 40, 300)
    bias_smooth = make_interp_spline(frames, biases, k=3)
    plt.plot(frames_new, bias_smooth(frames_new), '#B081D9', label="Head 0, Frame 0")
    plt.title("Head orientation 0 degrees")
    plt.xlabel("Frame orientation")
    plt.ylabel("Rod at PSE")
    plt.show()


def plotFigureBiasesAndVar():
    dataframe0 = pd.read_pickle("NoTiltPlotsAndData/data-Frame-01-06-frame-18-frames")
    dataframe15 = pd.read_pickle("Tilt15PlotsAndData/data-Frame-31-05-With-Const-tilt15")
    dataframe30 = pd.read_pickle("Tilt30PlotsAndData/data-Frame-01-06-tilt30")
    data = [dataframe0, dataframe15, dataframe30]
    datanames = ['Head at 0', "Head at 15", "Head at 30"]
    plt.subplots(2, 3)
    frames = [-45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40]
    for d, dataframe in enumerate(data):
        CWdata = dataframe[dataframe['pre_response'] == 1]
        CCWdata = dataframe[dataframe['pre_response'] == 0]

        CWdataPF = CWdata.groupby('frame_orientation').mean()
        CCWdataPF = CCWdata.groupby('frame_orientation').mean()

        plt.subplot(2, 3, d + 1)
        plt.plot(frames, CWdataPF['mu'], color="#77BAE4")
        plt.plot(frames, CCWdataPF['mu'], color="#E477AD")
        if d == 0:
            plt.ylabel("Bias ( mu )")
        plt.title(datanames[d])

        plt.subplot(2, 3, d + 4)
        plt.plot(frames, CWdataPF['sigma'], label=f"CW", color="#77BAE4")
        plt.plot(frames, CCWdataPF['sigma'], label=f"CCW", color="#E477AD")
        if d == 0:
            plt.ylabel("Variability ( sigma )")
        plt.xlabel("Frame orientation")

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


def anovaAnalysis():
    dataframe0 = pd.read_pickle("data-Frame-01-06-frame-10-frames")
    dataframe0free = pd.read_pickle("NoTiltPlotsAndData/data-Frame-01-06-frame-18-frames")
    dataframe15 = pd.read_pickle("Tilt15PlotsAndData/data-Frame-31-05-With-Const-tilt15")
    dataframe30 = pd.read_pickle("Tilt30PlotsAndData/data-Frame-01-06-tilt30")
    data = [dataframe0, dataframe0free, dataframe15, dataframe30]
    datanames = ['Head at 0 with Const', 'Head at 0 no Const', "Head at 15", "Head at 30"]

    for d, dataframe in enumerate(data):
        print(f"Analysis for {datanames[d]}")
        aov = AnovaRM(dataframe,
                      depvar='mu',
                      subject='nr',
                      within=['pre_response', 'frame_orientation']).fit()
        print(aov)

    masterDataSet = pd.concat(data[1:])
    condition = np.concatenate(([0] * 576, [15] * 576, [30] * 576))
    masterDataSet['condition'] = condition
    aov = AnovaRM(masterDataSet,
                  depvar='mu',
                  subject='nr',
                  within=['pre_response', 'frame_orientation', 'condition']).fit()
    print(aov)


def compare0():
    dataframe0 = pd.read_pickle("data-Frame-01-06-frame-10-frames")
    dataframe0free = pd.read_pickle("NoTiltPlotsAndData/data-Frame-01-06-frame-18-frames")
    data = [dataframe0, dataframe0free]
    datanames = ['Head at 0 with const', 'Head at 0 no const']
    plt.subplots(2, 1)
    framesConst = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    framesNoConst = [-45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40]
    frames = [framesConst, framesNoConst]
    for d, dataframe in enumerate(data):
        CWdata = dataframe[dataframe['pre_response'] == 1]
        CCWdata = dataframe[dataframe['pre_response'] == 0]

        CWdataPF = CWdata.groupby('frame_orientation').mean()
        CCWdataPF = CCWdata.groupby('frame_orientation').mean()

        plt.subplot(2, 1, 1)
        plt.plot(frames[d], CWdataPF['mu'], label=f"CW")#, color="#77BAE4")
        plt.plot(frames[d], CCWdataPF['mu'], label=f"CCW")#, color="#E477AD")
        if d == 0:
            plt.ylabel("Bias ( mu )")
        plt.title(datanames[d])

        plt.subplot(2, 1, 2)
        plt.plot(frames[d], CWdataPF['sigma'], label=f"CW")#, color="#77BAE4")
        plt.plot(frames[d], CCWdataPF['sigma'], label=f"CCW")#, color="#E477AD")
        if d == 0:
            plt.ylabel("Variability ( sigma )")
        plt.xlabel("Frame orientation")

    plt.legend(prop={'size': 8})
    plt.tight_layout()
    plt.show()


compare0()
###### TEST STUFF, PRINTING PROBS FOR TEST #######
# data = pd.read_csv('Controls/c7/c7_frame.txt', skiprows=13, sep=" ")
# data.drop('reactionTime', inplace=True, axis=1)
# data.drop('Unnamed: 4', inplace=True, axis=1)
# model00 = VisualVestibularVerticalityModel([0, 0, 6.5, 0.07, 2.21, 4.87, 52.26, 0.8, 0.1])
# model300 = VisualVestibularVerticalityModel([30, 0, 6.5, 0.07, 2.21, 4.87, 52.26, 0.8, 0.1])
# model020 = VisualVestibularVerticalityModel([0, 20, 6.5, 0.07, 2.21, 4.87, 52.26, 0.8, 0.1])
# model3020 = VisualVestibularVerticalityModel([30, -20, 6.5, 0.07, 2.21, 4.87, 52.26, 0.8, 0.1])
#
# matrix = createMatrixProb(data)
# rod_orients_all = np.sort(data.rodOri.unique())
# frame_orients_all = np.sort(data.frameOri.unique())
# rods = np.linspace(-15, 15, 101)
#
# for frame in frame_orients_all :
#     cur_mod = VisualVestibularVerticalityModel([0, frame, 6.5, 0.07, 2.21, 4.87, 52.26, 0.8, 0.1])
#     plt.plot(rods, cur_mod.getCWProbability(rods))
#     print(np.where(frame_orients_all == -20))
#     plt.plot(rod_orients_all, matrix[(np.where(frame_orients_all == -20)[0][0])], 'ro')
#     plt.xlabel("Rod orientations")
#     plt.ylabel("P(CW | rod)")
#     plt.title(f"Head 0, Frame {frame}")
#     plt.show()
