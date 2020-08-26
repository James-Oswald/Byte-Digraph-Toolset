
import os
import sys
import cv2
import numpy as np

os.chdir(sys.path[0])

dataFileName = "conti.c"

def fileToByteArray(fileName):
    with open(fileName, mode="rb") as file:
        return file.read()

def BDM(data):
    M = np.zeros((256, 256))
    for i in range(0, len(data) - 1):
        M[data[i], data[i+1]] = 1
    return M

def BDM2PNG(bdm):
    imageData = np.tile(np.expand_dims(np.where(bdm == 1, 255, bdm), axis=2), (1, 1, 3))
    cv2.imwrite("../images/" + dataFileName + ".BDM.png", imageData)

BDM2PNG(BDM(fileToByteArray("../data/" + dataFileName)))