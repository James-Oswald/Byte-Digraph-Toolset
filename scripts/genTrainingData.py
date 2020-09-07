
import os
import sys
import cv2
import numpy as np
import random as r
from BDM import * 

os.chdir(sys.path[0])

#generate training images
imSize = 512
numImgs = 100
img = np.zeros((imSize, imSize, 3), np.uint8)
for i in range(0, 50):
    cv2.rectangle(img, (r.randint(0, imSize), r.randint(0, imSize)), (r.randint(0, imSize), r.randint(0, imSize)), (r.randint(0, 255), r.randint(0, 255), r.randint(0, 255)), r.randint(3, 7))
    cv2.line(img, (r.randint(0, imSize), r.randint(0, imSize)), (r.randint(0, imSize), r.randint(0, imSize)), (r.randint(0, 255), r.randint(0, 255), r.randint(0, 255)), r.randint(3, 7))
for i in range(10, imSize, 5):
    point = [r.randint(0, imSize - i), r.randint(0, imSize - i)]
    cv2.imwrite("../data/trainingData/recs" + str(i) + ".png", img[point[0]: point[0] + i, point[1]: point[1] + i, :])
cutPath = "../data/trainingData/"
for i in os.listdir(cutPath):
    BDM2PNG(BDM(fileToByteArray(cutPath + i)), "../data/trainingDataBDM/" + i + ".BDM.png")

#generate training text
