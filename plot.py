import matplotlib.pyplot as plt

from VisualVestibularVerticalityModel import VisualVestibularVerticalityModel

Frame = 0
Head = -30
model = VisualVestibularVerticalityModel(Head, Frame)
model.plot(0)

Frame = 0
Head = 30
model = VisualVestibularVerticalityModel(Head, Frame)
model.plot(0)


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
