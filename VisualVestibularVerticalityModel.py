import matplotlib.pyplot as plt
import numpy as np
import math

from scipy.interpolate import interp1d
from scipy.stats import norm, vonmises


def sig2kap(sigma):
    """
    This is a helper function for contextual likelihood

    Generally people find sigma values easier to understand from kappa values.
    Thus, in the paper for discussion sigma variability is used instead of kappa.
    The formula below is based on formula used by Niels
    :param sigma: a number
    :return: kappa : a number
    """
    return 3994.5 / (np.power(sigma, 2) + 22.6)


class VisualVestibularVerticalityModel:

    def __init__(self, params):
        self.head_space_orients = np.linspace(-180, 180, 361)

        self.H_true = params[0]
        self.F_true = params[1]

        # for the prior
        self.sigma_hp = params[2]

        # for the vestibular likelihood
        self.alpha_hs = params[3]
        self.beta_hs = params[4]

        # for the contextual likelihood
        self.k_ver = sig2kap(params[5])
        self.k_hor = sig2kap(params[6])
        self.tau = params[7]

        # for the transformation
        self.A_ocr = 14.6

        self.head_space_prior = np.zeros(361)
        self.vest_likelihood = np.zeros(361)
        self.cont_likelihood = np.zeros(361)
        self.head_space_post = np.zeros(361)

        self.head_space_post_cdf_interp = None
        self.head_space_post_interp = None

    def head_in_space_prior(self):
        """ Function computing the head in space prior

        This function is in degrees
        Based on information from page 4 we assume that the brain uses prior knowledge
        that our head id usually upright, thus at 0 degrees
        """
        self.head_space_prior = norm.pdf(self.head_space_orients, 0, self.sigma_hp)

    def vestibular_likelihood(self):
        """ Function computing the vestibular likelihood

        This function is in degrees.
        Based on formula (2) the variability sigma_hs is computed.
        """
        sigma_hs = self.alpha_hs * abs(self.H_true) + self.beta_hs
        self.vest_likelihood = norm.pdf(self.head_space_orients, self.H_true, sigma_hs)

    def contextual_likelihood(self):
        """ Function computing contextual likelihood

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

        for i in range(0, 4):
            # here if we have a frame at 20 degrees, then the observed is flipped,
            # so -20, so we would have a peak at -20 instead of 20, so I will flip it from now.
            vonmisesX = phi[i] - obsr_frame_orient - HSO_rad
            self.cont_likelihood += vonmises.pdf(vonmisesX, kappa[i])

        self.cont_likelihood = np.degrees(self.cont_likelihood)

    def head_in_space_posterior(self):
        """ Function calculating the head in space posterior

        This function is in degrees
        The posterior is obtained from multiplication of the likelihoods with the prior and
        it is subsequently normalized
        """
        self.head_in_space_prior()
        self.vestibular_likelihood()
        self.contextual_likelihood()

        self.head_space_post = self.cont_likelihood * self.vest_likelihood * self.head_space_prior
        self.head_space_post = self.head_space_post / np.sum(self.head_space_post)

        self.head_space_post_interp = interp1d(self.head_space_orients, self.head_space_post)
        self.head_space_post_cdf_interp = interp1d(self.head_space_orients, np.cumsum(self.head_space_post))

    def getCWProbability(self, rod_orient):
        """
        This function calculates the probability of a clockwise response for a given rod
        from the head-in-space posterior

        :param rod_orient: a single number
        :return:
        """
        self.head_in_space_posterior()

        fromWhereToSum = self.H_true - rod_orient
        rod_prob = 1 - self.head_space_post_cdf_interp(fromWhereToSum)
        return rod_prob

    def plot(self):
        self.head_in_space_posterior()

        plt.subplots(2, 2)

        plt.subplot(2, 2, 1)
        plt.plot(self.head_space_orients, self.head_space_prior)
        plt.xlim(- 20, 20)
        plt.title("Head-in-space prior")

        plt.subplot(2, 2, 2)
        plt.plot(self.head_space_orients, self.vest_likelihood)
        plt.xlim(self.H_true - 20, self.H_true + 20)
        plt.title(f"Vestibular likelihood for Head {self.H_true}")

        plt.subplot(2, 2, 3)
        plt.plot(self.head_space_orients, self.cont_likelihood)
        plt.title(f"Contextual likelihood for Frame {self.F_true}")

        plt.subplot(2, 2, 4)
        plt.plot(self.head_space_orients, self.head_space_post)
        plt.xlim(self.H_true - 20, self.H_true + 20)
        plt.title("Head-in-space posterior")
        plt.tight_layout()
        plt.show()

        rods = np.linspace(-15, 15, 101)
        plt.plot(rods, self.getCWProbability(rods))
        plt.xlabel("Rod orientations")
        plt.ylabel("P(CW | rod)")
        plt.show()
