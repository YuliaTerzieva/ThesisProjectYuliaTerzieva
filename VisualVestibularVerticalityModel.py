import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import norm, vonmises


class VisualVestibularVerticalityModel:

    def __init__(self, H_true, F_true):
        self.rod_orients = np.array([-7, -4, -2, -1, 0, 1, 2, 4, 7])
        self.frame_orients = np.array(np.linspace(-45, 40, 18))
        self.head_space_orients = np.linspace(-180, 180, 360)

        self.H_true = H_true
        self.F_true = F_true
        self.sigma_hp = 6.5
        self.alpha_hs = 0.07
        self.beta_hs = 2.21
        self.k_ver = 86.24
        self.k_hor = 1.451
        self.tau = 0.8
        self.lapse_rate = 0.02
        self.A_ocr = 14.6

        self.contextual_likelihood()
        self.vestibular_likelihood()
        self.head_in_space_prior()
        self.head_in_space_posterior()

    def contextual_likelihood(self):
        obsr_frame_orient = math.radians(- (self.F_true - self.H_true) - self.A_ocr * math.sin(abs(self.H_true)))
        kappa_1 = self.k_ver - (1 - math.cos(abs(2 * obsr_frame_orient))) * self.tau * (self.k_ver - self.k_hor)
        kappa_2 = self.k_hor + (1 - math.cos(abs(2 * obsr_frame_orient))) * (1 - self.tau) * (self.k_ver - self.k_hor)
        kappa = [kappa_1, kappa_2, kappa_1, kappa_2]
        phi = np.radians(np.array([0, 90, 180, 270]))

        self.cont_likelihood = np.zeros(360)
        for i in range(0, 4):
            vonmisesX = obsr_frame_orient + phi[i] - np.radians(self.head_space_orients)
            self.cont_likelihood += vonmises.pdf(vonmisesX, kappa[i])

        self.cont_likelihood = np.degrees(self.cont_likelihood)

    def vestibular_likelihood(self):
        self.sigma_hs = self.alpha_hs * abs(self.H_true) + self.beta_hs
        self.vest_likelihood = norm.pdf(self.head_space_orients, self.H_true, self.sigma_hs)

    def head_in_space_prior(self):
        self.head_space_prior = norm.pdf(self.head_space_orients, 0, self.sigma_hp)

    def head_in_space_posterior(self):
        self.head_space_post = self.cont_likelihood * self.vest_likelihood * self.head_space_prior

    def getRodProbability(self):

        MAP = self.head_space_post.argmax()
        rod_in_space = np.zeros(len(self.rod_orients))
        for i, rod_orient in enumerate(self.rod_orients):
            # Eye-in-head orientation transformation - Eye torsion + line-on-eye = Line_on_retina
            line_on_retina = -(rod_orient - self.H_true) - self.A_ocr * math.sin(abs(self.H_true))
            rod_in_space[i] = MAP + line_on_retina

        rod_in_space_cdf = np.cumsum(rod_in_space) / np.sum(rod_in_space)
        PCW = self.lapse_rate + (1 - 2*self.lapse_rate) * rod_in_space_cdf
        return PCW
