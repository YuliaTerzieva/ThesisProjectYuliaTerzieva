import matplotlib.pyplot as plt
import numpy as np

from VisualVestibularVerticalityModel import VisualVestibularVerticalityModel






Frame = -10
Head = 30
model = VisualVestibularVerticalityModel(Head, Frame)
x = model.head_space_orients
fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(x, model.head_space_prior)
axs[0, 0].set_title("Prior")
axs[0, 1].plot(x, model.vest_likelihood)
axs[0, 1].set_title("Vestibular Likelihood")
axs[1, 0].plot(x, model.cont_likelihood)
axs[1, 0].set_title("Contextual Posterior")
axs[1, 1].plot(x, model.head_space_post)
axs[1, 1].set_title("Posterior")
fig.tight_layout()
plt.show()

model.getRodProbability(0)
# rod_orients = np.array([-7, -4, -2, -1, 0, 1, 2, 4, 7])
# # PCW = np.zeros(len(rod_orients))
# # for i, r in enumerate(rod_orients):
# #     PCW[i] = model.getRodProbability(r)
#
# # plt.show()
# PCW = model.getRodProbability()
# plt.plot(rod_orients, PCW)
# plt.xlabel("Rod orientation")
# plt.ylabel("P(CW)")
# plt.title("Probability of answering CW given \n Frame orientation %d and Head orientation %d" % (Frame, Head ))
# plt.show()
