import os
import sys
os.chdir(sys.path[0])
from BDM import *

from tensorflow import keras
import numpy as np

test = [0, 1001, 2002]

x_train = np.load("../data/procTrainingData/data.npz")["data"]
for i in test:
    BDM2PNG(x_train[i, :, :], "../data/BDMpngs/" + str(i) + ".png")

x_train = np.expand_dims(x_train, -1)   #This is the color channels dim
x_train = np.expand_dims(x_train, 1)    #This is the "batch size" dim
print(x_train.shape)

model = keras.models.load_model("../models/BBDMModel")
for i in test:
    print(model(x_train[i, :, :, :, :]))