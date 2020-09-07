
import os
import sys
import cv2
import numpy as np
import random as r
from BDM import * 

os.chdir(sys.path[0])

imagesPerClass = 5000

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
        cv2.imwrite("../data/trainingData/recPng" + str(j) + ".png", img[point[0]: point[0] + i, point[1]: point[1] + i, :])
        cv2.imwrite("../data/trainingData/recJpg" + str(j) + ".jpg", img[point[0]: point[0] + i, point[1]: point[1] + i, :])
        print("Generated Image " + str(j))
    
#generate training text
def genTxts():
    textData = [fileToByteArray("../data/TheRepublic.txt"), fileToByteArray("../data/OliverTwist.txt")]
    for j in range(0, imagesPerClass):
        slected = r.choice(textData)
        strlen = r.randint(100, 2000)
        start = r.randint(0, len(slected) - strlen)
        end = start + strlen
        f = open("../data/trainingData/text" + str(j) + ".txt", "wb")
        f.write(slected[start:end])
        f.close()
        print("Generated Text " + str(j))

def generateBDMs():
    cutPath = "../data/trainingData/"
    for i in os.listdir(cutPath):
        BDM2IMG(BDM(fileToByteArray(cutPath + i)), "../data/trainingDataBDM/" + i + ".BDM.jpg")
        print("Generated BDM of " + i)

genTxts()
genImgs()
generateBDMs()