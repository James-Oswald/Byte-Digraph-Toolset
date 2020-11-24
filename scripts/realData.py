
#import os
import numpy as np
import tensorflow_datasets as tfds
from tensorflow import make_ndarray

from BDM import BDM, fileToByteArray


dataDir = "D:\\Datasets"
rawTrainingDataPath = "../data/rawTrainingData"



def generateAudioBDM():
    gtzan = tfds.load("gtzan", data_dir=dataDir)
    print(make_ndarray(gtzan.take(1).audio))
    #gtzanMS = 

generateAudioBDM()

#def generateImagesBDM():


#def generateTextsBDM():



'''def generateBDMDataFile():
    fileNames = os.listdir(rawTrainingDataPath)
    trainingData = np.zeros((len(fileNames), 256, 256))
    trainingLabels = np.zeros((len(fileNames)))
    for i in range(len(fileNames)):
        trainingData[i, :, :] = BDM(fileToByteArray(rawTrainingDataPath + "/" + fileNames[i]))
        trainingLabels[i] = ord(fileNames[i][0])
        print("Generated BDM:" + str(i))
    np.savez("../data/procTrainingData/data.npz", labels=trainingLabels, data=trainingData)

generateBDMDataFile()'''