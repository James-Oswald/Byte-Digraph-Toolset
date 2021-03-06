
import os
import sys
import cv2
import numpy as np

os.chdir(sys.path[0])

#dataFileName = ""

def fileToByteArray(fileName):
    with open(fileName, mode="rb") as file:
        return file.read()

def BBDM(data):
    M = np.zeros((256, 256))
    for i in range(0, len(data) - 1):
        M[data[i], data[i+1]] = 1
    return M

def PDBDM(data):
    M = np.zeros((256, 256))
    for i in range(0, len(data) - 1):
        M[data[i], data[i+1]] += 1
    return M/M.max()

def BDM2File(bdm, filename):
    np.save(filename, bdm)

def File2BDM(filename):
    return np.load(filename)

def BDM2PNG(bdm, filename):
    imageData = np.tile(np.expand_dims(np.where(bdm == 1, 255, bdm), axis=2), (1, 1, 3))
    cv2.imwrite(filename, imageData)

def BDM2IMG(bdm, path):
    imageData = np.tile(np.expand_dims(np.where(bdm == 1, 255, bdm), axis=2), (1, 1, 3))
    cv2.imwrite(path, imageData)


#BDM2PNG(BDM(fileToByteArray("../data/" + dataFileName)))