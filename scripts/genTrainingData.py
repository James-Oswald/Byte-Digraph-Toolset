
import os
import sys
import cv2
import numpy as np
import random as r
from BDM import * 

os.chdir(sys.path[0])

rawTrainingDataPath = "../data/rawTrainingData"
imagesPerClass = 1000

#generate training images
def genImgs():
    imSize = 512
    img = np.zeros((imSize, imSize, 3), np.uint8)
    for i in range(0, 50):
        cv2.rectangle(img, (r.randint(0, imSize), r.randint(0, imSize)), (r.randint(0, imSize), r.randint(0, imSize)), (r.randint(0, 255), r.randint(0, 255), r.randint(0, 255)), r.randint(3, 7))
        cv2.line(img, (r.randint(0, imSize), r.randint(0, imSize)), (r.randint(0, imSize), r.randint(0, imSize)), (r.randint(0, 255), r.randint(0, 255), r.randint(0, 255)), r.randint(3, 7))
    for j in range(0, imagesPerClass):
        i = r.randint(10, imSize - 10)
        point = [r.randint(0, imSize - i), r.randint(0, imSize - i)]
        cv2.imwrite(rawTrainingDataPath + "/p." + str(j) + ".png", img[point[0]: point[0] + i, point[1]: point[1] + i, :])
        #cv2.imwrite(rawTrainingDataPath + "/j." + str(j) + ".jpg", img[point[0]: point[0] + i, point[1]: point[1] + i, :])
        print("Generated Image " + str(j))
    
#generate training text
def genTxts():
    textData = [fileToByteArray("../data/sampleData/TheRepublic.txt"), fileToByteArray("../data/sampleData/OliverTwist.txt")]
    for j in range(0, imagesPerClass):
        slected = r.choice(textData)
        strlen = r.randint(100, 2000)
        start = r.randint(0, len(slected) - strlen)
        end = start + strlen
        f = open(rawTrainingDataPath + "/t." + str(j) + ".txt", "wb")
        f.write(slected[start:end])
        f.close()
        print("Generated Text " + str(j))

def genExes():
    textData = [fileToByteArray("../data/sampleData/opengl32.dll"), fileToByteArray("../data/sampleData/conti.exe")]
    for j in range(0, imagesPerClass):
        slected = r.choice(textData)
        strlen = r.randint(100, 2000)
        start = r.randint(0, len(slected) - strlen)
        end = start + strlen
        f = open(rawTrainingDataPath + "/x." + str(j) + ".txt", "wb")
        f.write(slected[start:end])
        f.close()
        print("Generated x86 " + str(j))

#def generateBDMs():
#    for i in os.listdir(rawTrainingDataPath):
#        BDM2PNG(BDM(fileToByteArray(rawTrainingDataPath + "/" + i)), "../data/trainingDataBDM/" + bdm + ".BDM.jpg")
#        print("Generated BDM of " + i)

def generateBDMDataFile():
    fileNames = os.listdir(rawTrainingDataPath)
    trainingData = np.zeros((len(fileNames), 256, 256))
    trainingLabels = np.zeros((len(fileNames)))
    for i in range(len(fileNames)):
        trainingData[i, :, :] = BDM(fileToByteArray(rawTrainingDataPath + "/" + fileNames[i]))
        trainingLabels[i] = ord(fileNames[i][0])
        print("Generated BDM:" + i)
    np.savez("../data/procTrainingData/data.npz", labels=trainingLabels, data=trainingData)    
    
genTxts()
genImgs()
genExes()
generateBDMDataFile()