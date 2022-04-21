import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import norm, vonmises


def sig2kap(sigma):
    """ This is a helper function for contextual likelihood

    Generally people find sigma values easier to understand from kappa values.
    Thus, in the paper for discussion sigma variability is used instead of kappa.
    The formula below is based on formula used by Niels
    """
    return 3994.5 / (np.power(sigma, 2)+22.6)

class VisualVestibularVerticalityModel:
    """ ToDo
    """

    def __init__(self, H_true, F_true):
        """ ToDo
        """
        self.rod_orients = np.array([-7, -4, -2, -1, 0, 1, 2, 4, 7])
        self.frame_orients = np.array(np.linspace(-45, 40, 18))
        self.head_space_orients = np.linspace(-180, 180, 360)

        # The following parameters are based on values from the paper Table 1 page 7
        self.H_true = H_true
        self.F_true = F_true
        self.sigma_hp = 6.5
        self.alpha_hs = 0.07
        self.beta_hs = 2.21
        self.k_ver = sig2kap(4.87)
        self.k_hor = sig2kap(52.26)
        self.tau = 0.8
        self.lapse_rate = 0.02
        self.A_ocr = 14.6

        self.head_space_prior = None
        self.vest_likelihood = None
        self.cont_likelihood = None
        self.head_space_post = None

        self.contextual_likelihood()
        self.vestibular_likelihood()
        self.head_in_space_prior()
        self.head_in_space_posterior()

    def contextual_likelihood(self):
        """This is the summary line

        This function is in radians.
        The VonMisses function works with radians, thus we initially convert all necessary variables
        from degrees to radians and then compute the likelihood. Before the function is done the resultant
        likelihood is again transformed into degrees
        """
        F_rad = np.radians(self.F_true)
        H_rad = np.radians(self.H_true)
        HSO_rad = np.radians(self.head_space_orients)
        A_ocr_rad = np.radians(self.A_ocr)

        obsr_frame_orient = - (F_rad - H_rad) - A_ocr_rad * math.sin(abs(H_rad))
        kappa_1 = self.k_ver - (1 - math.cos(abs(2 * obsr_frame_orient))) * self.tau * (self.k_ver - self.k_hor)
        kappa_2 = self.k_hor + (1 - math.cos(abs(2 * obsr_frame_orient))) * (1 - self.tau) * (self.k_ver - self.k_hor)
        kappa = [kappa_1, kappa_2, kappa_1, kappa_2]
        phi = np.radians(np.array([0, 90, 180, 270]))

        self.cont_likelihood = np.zeros(360)
        for i in range(0, 4):
            vonmisesX = obsr_frame_orient + phi[i] - HSO_rad
            self.cont_likelihood += vonmises.pdf(vonmisesX, kappa[i])

        self.cont_likelihood = np.degrees(self.cont_likelihood)

    def vestibular_likelihood(self):
        """ Function computing the vestibular likelihood

        This function is in degrees.
        Based on formula (2) the variability sigma_hs is computed.
        """
        sigma_hs = self.alpha_hs * abs(self.H_true) + self.beta_hs
        self.vest_likelihood = norm.pdf(self.head_space_orients, self.H_true, sigma_hs)

    def head_in_space_prior(self):
        """ Function computing the head in space prior

        This function is in degrees
        Based on information from page 4 we assume that the brain uses prior knowledge
        that our head id usually upright
        """
        self.head_space_prior = norm.pdf(self.head_space_orients, 0, self.sigma_hp)

    def head_in_space_posterior(self):
        """ Function calculating the head in space posterior

        This function is in degrees
        The posterior is obtained from multiplication of the likelihoods with the prior and
        it is subsequently normalized
        """
        self.head_space_post = self.cont_likelihood * self.vest_likelihood * self.head_space_prior
        self.head_space_post = self.head_space_post / np.sum(self.head_space_post)

    def getRodProbability(self, rod_orient):

        line_on_retina = -(rod_orient - self.H_true) - self.A_ocr * math.sin(abs(self.H_true))
        print("Line on retine", line_on_retina)
        line_in_space_X = self.head_space_orients - line_on_retina

        plt.plot(line_in_space_X, self.head_space_post)
        # plt.title()
        plt.show()

        plt.plot(line_in_space_X, np.cumsum(self.head_space_post))
        plt.show()
        # Todo: 1) Fix the code above - instead of moving the x-axis -> move the value array
        #       2) Transform into cdf - done but check
        #       3) Remove unnecessary code

        # rod_on_retina = -(R_true - self.H_true) - self.A_ocr * math.sin(abs(self.H_true))
        # normalized = self.head_space_post / np.sum(self.head_space_post)
        # MAP = self.head_space_orients[normalized.argmax()]
        #
        # return MAP - rod_on_retina
        # rod_in_space = self.head_space_post - rod_on_retina
        # # rod_in_space = rod_in_space / np.sum(rod_in_space)
        # plt.plot(self.head_space_orients, rod_in_space)
        # plt.title("Test")
        # return self.head_space_orients[rod_in_space.argmax()]

        # MAP = self.head_space_orients[self.head_space_post.argmax()]
        # print("MAP is ", MAP)
        #
        # rod_in_space = np.zeros(len(self.rod_orients))
        # for i, rod_orient in enumerate(self.rod_orients):
        #     # Eye-in-head orientation transformation - Eye torsion + line-on-eye = Line_on_retina
        #     line_on_retina = -(rod_orient - self.H_true) - self.A_ocr * math.sin(abs(self.H_true))
        #     rod_in_space[i] = MAP - line_on_retina
        #
        # # print("Rod in space", rod_in_space)
        # rod_in_space_cdf = np.cumsum(rod_in_space) / np.sum(rod_in_space)
        # PCW = self.lapse_rate + (1 - 2*self.lapse_rate) * rod_in_space_cdf
        # return PCW
