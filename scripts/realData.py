
import numpy as np
import tarfile
import random

from BDM import BDM

dataDir = "D:/Datasets/MyTars/"
bytesPerSample = 10
samplesPerClass = 1000
classes = ["audio", "text", "exedata", "image"]

datasetIndex = 0
trainingData = trainingData = np.zeros((samplesPerClass * len(classes), 256, 256))
trainingLabels = [None] * (samplesPerClass * len(classes))

for file in classes:
    datasetTar = tarfile.open(dataDir + file + ".tar")
    datafiles = datasetTar.getmembers()
    sfl = samplesPerClass // len(datafiles)
    sml = samplesPerClass % len(datafiles)
    numExamples = 0
    fileNumber = 0
    while numExamples < samplesPerClass:
        print("Generating " + file + " samples #" + str(numExamples)) 
        numSamples = sfl + (1 if fileNumber < sml else 0)
        fileBytes = bytearray(datasetTar.extractfile(datafiles[fileNumber]).read())
        if bytesPerSample > len(fileBytes):
            raise Exception("bytesPerSample is %d but len(fileBytes) of %s is only %d" % (bytesPerSample, datafiles[fileNumber].name, len(fileBytes)))
        for i in range(numSamples):
            startIndex = random.randint(0, len(fileBytes) - bytesPerSample)
            trainingData[datasetIndex, :, :] = BDM(fileBytes[startIndex:(startIndex + bytesPerSample)])
            trainingLabels[datasetIndex] = file
            datasetIndex += 1
        numExamples += numSamples
        fileNumber += 1

np.savez("../data/procTrainingData/realData.npz", labels=trainingLabels, data=trainingData) 