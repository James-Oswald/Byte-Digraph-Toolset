import os
import sys
import cv2
import numpy as np
import math

os.chdir(sys.path[0])
bytesPerLine = 32
dataFileName = "conti.c"
fileData = np.fromfile("../data/" + dataFileName, dtype="uint8")
paddedFileData = np.pad(fileData, (0, bytesPerLine - (np.size(fileData) % bytesPerLine)))
data = np.reshape(paddedFileData, (bytesPerLine, np.size(paddedFileData) // bytesPerLine))
data = np.expand_dims(data, axis=2)
imageData = np.tile(data, (1, 1, 3))
cv2.imwrite("../images/" + dataFileName + ".png", imageData)