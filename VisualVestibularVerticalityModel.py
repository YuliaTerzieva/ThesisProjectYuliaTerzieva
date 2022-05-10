import matplotlib.pyplot as plt
import numpy as np
import math
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

    def __init__(self, H_true, F_true):
        self.rod_orients = np.array([-7, -4, -2, -1, 0, 1, 2, 4, 7])
        self.frame_orients = np.array(np.linspace(-45, 40, 18))
        self.head_space_orients = self.rod_space_orients = np.linspace(-180, 180, 360)

        self.H_true = H_true
        self.F_true = F_true

        # The following parameters are based on values from the paper Table 1 page 7

        # for the prior
        self.sigma_hp = 6.5

        # for the vestibular likelihood
        self.alpha_hs = 0.07
        self.beta_hs = 2.21

        # for the contextual likelihood
        self.k_ver = sig2kap(4.87)
        self.k_hor = sig2kap(52.26)
        self.tau = 0.8

        # for the transformation
        self.A_ocr = 14.6

        self.head_space_prior = np.zeros(360)
        self.vest_likelihood = np.zeros(360)
        self.cont_likelihood = np.zeros(360)
        self.head_space_post = np.zeros(360)
        self.rod_space_prob = np.zeros(360)

    def head_in_space_prior(self):
        """ Function computing the head in space prior

        This function is in degrees
        Based on information from page 4 we assume that the brain uses prior knowledge
        that our head id usually upright, thus at 0 degrees
        """
        self.head_space_prior = norm.pdf(self.head_space_orients, 0, self.sigma_hp)
        self.head_space_prior = self.head_space_prior / np.sum(self.head_space_prior)

    def vestibular_likelihood(self):
        """ Function computing the vestibular likelihood

        This function is in degrees.
        Based on formula (2) the variability sigma_hs is computed.
        """
        sigma_hs = self.alpha_hs * abs(self.H_true) + self.beta_hs
        self.vest_likelihood = norm.pdf(self.head_space_orients, self.H_true, sigma_hs)
        self.vest_likelihood = self.vest_likelihood / np.sum(self.vest_likelihood)


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

        for i in range(0, 4):
            vonmisesX = obsr_frame_orient + phi[i] - HSO_rad
            self.cont_likelihood += vonmises.pdf(vonmisesX, kappa[i])

        self.cont_likelihood = np.degrees(self.cont_likelihood)
        self.cont_likelihood = self.cont_likelihood / np.sum(self.cont_likelihood)


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

    def getRodProbability(self, rod_orient):

        self.head_in_space_posterior()
        # For the computation of rod on retina I'm using formula 7
        rod_on_retina = int(self.H_true - rod_orient - self.A_ocr * math.sin(abs(self.H_true)))

        print(f" we have {self.H_true - rod_orient} + {- self.A_ocr * math.sin(abs(self.H_true))}")
        print(f" the rod on retina is {rod_on_retina}")

        self.rod_space_prob = np.append(self.head_space_post[rod_on_retina:], self.head_space_post[:rod_on_retina])


    def plot(self, rod_orient):
        self.getRodProbability(rod_orient)

        plt.subplots(3, 2)

        plt.subplot(3, 2, 1)
        plt.plot(self.head_space_orients, self.head_space_prior)
        plt.title("Head-in-space prior")

        plt.subplot(3, 2, 2)
        plt.plot(self.head_space_orients, self.vest_likelihood)
        plt.title(f"Vestibular likelihood for Head {self.H_true}")

        plt.subplot(3, 2, 3)
        plt.plot(self.head_space_orients, self.cont_likelihood)
        plt.title(f"Contextual likelihood for Frame {self.F_true}")

        plt.subplot(3, 2, 4)
        plt.plot(self.head_space_orients, self.head_space_post)
        plt.title("Head-in-space posterior")

        plt.subplot(3, 2, 5)
        plt.plot(self.rod_space_orients, self.rod_space_prob)
        plt.title("Rod in space probability estimation")

        plt.subplot(3, 2, 6)
        plt.plot(self.head_space_orients, np.cumsum(self.rod_space_prob),
                 label=f"value at x at 0 = ")
        plt.title(f"Probability of CW response given \n H = {self.H_true}, F = {self.F_true} for rod = {rod_orient}")
        plt.legend()

        plt.tight_layout()
        plt.show()
