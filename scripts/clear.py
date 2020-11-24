
import os
import sys

os.chdir(sys.path[0])
paths = ["../data/rawTrainingData/", "../data/BDMpngs/"]
for path in paths:
    for file in os.listdir(path):
        os.remove(path + file)
