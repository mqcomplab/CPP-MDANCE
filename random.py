import numpy as np

data=np.load("./tests/data/backbone.npy")
np.savetxt("./tests/data/backbone.csv", data, delimiter=",")