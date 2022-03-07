# This is a sample Python script.
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import norm, vonmises
from scipy.signal import peak_widths, find_peaks

rod_orients = np.array([-7, -4, -2, -1, 0, 1, 2, 4, 7])
frame_orients = np.array(np.linspace(-45, 40, 18))
head_space_orients = np.linspace(-180, 180, 360)
H_true = 0
F_true = 0
sigma_hp = 6.5
alpha_hs = 0.07
beta_hs = 2.21
k_ver = 4.87
k_hor = 52.26
tau = 0.8
lapse_rate = 0.02
A_ocr = 14.6

# # Contextual Likelihood
# obsr_frame_orient = - (F_true - H_true) - A_ocr * math.sin(abs(H_true))
# kappa_1 = k_ver - (1 - math.cos(abs(2 * obsr_frame_orient))) * tau * (k_ver - k_hor)
# kappa_2 = k_hor + (1 - math.cos(abs(2 * obsr_frame_orient))) * (1 - tau) * (k_ver - k_hor)
# kappa = [kappa_1, kappa_2, kappa_1, kappa_2]
# phi = [0, 90, 180, 270]
#
# vonmisesX = frame_orients + phi[0] - H_true
# cont_likelihood = vonmises.pdf(vonmisesX, kappa[0])
# plt.plot(frame_orients, cont_likelihood)
# plt.show()
#
# for i in range(1, 4):
#     vonmisesX = frame_orients + phi[i] - H_true
#     cont_likelihood += vonmises.pdf(vonmisesX, kappa[i])
#
# plt.plot(frame_orients, cont_likelihood)
# plt.show()

# Contextual Likelihood
obsr_frame_orient = - (F_true - H_true) - A_ocr * math.sin(abs(H_true))
kappa_1 = k_ver - (1 - math.cos(abs(2 * obsr_frame_orient))) * tau * (k_ver - k_hor)
kappa_2 = k_hor + (1 - math.cos(abs(2 * obsr_frame_orient))) * (1 - tau) * (k_ver - k_hor)
kappa = [kappa_1, kappa_2, kappa_1, kappa_2]
phi = [0, 90, 180, 270]

vonmisesX = obsr_frame_orient + phi[0] - head_space_orients
cont_likelihood = vonmises.pdf(vonmisesX, kappa[0])
plt.plot(head_space_orients, cont_likelihood)
plt.show()

for i in range(1, 4):
    vonmisesX = obsr_frame_orient + phi[i] - head_space_orients
    cont_likelihood += vonmises.pdf(vonmisesX, kappa[i])

plt.plot(head_space_orients, cont_likelihood)
plt.show()

# Vestibular Likelihood
sigma_hs = alpha_hs * abs(H_true) + beta_hs
vest_likelihood = norm.pdf(head_space_orients, H_true, sigma_hs)

plt.plot(head_space_orients, vest_likelihood)
plt.show()

# Head-in-space prior
head_space_prior = norm.pdf(head_space_orients, 0, sigma_hp)

plt.plot(head_space_orients, head_space_prior)
plt.show()


# Head-in-space posterior
head_space_post = cont_likelihood * vest_likelihood * head_space_prior

plt.plot(head_space_orients, head_space_post)
plt.show()
