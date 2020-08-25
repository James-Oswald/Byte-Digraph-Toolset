import os
import sys
import cv2
import numpy as np
import math

def calcEntropy(data, index, rangeSize):
    start = index - rangeSize
    end = index + rangeSize
    for i in data[start:end]:
        hist[i] = hist.get(i, 0) + 1
    entropy = 0
     z

def entropy(data, blocksize, offset, symbols=256):
    if len(data) < blocksize:
        raise ValueError, "Data length must be larger than block size."
    if offset < blocksize/2:
        start = 0
    elif offset > len(data)-blocksize/2:
        start = len(data)-blocksize/2
    else:
        start = offset-blocksize/2
    hist = {}
    for i in data[start:start+blocksize]:
        hist[i] = hist.get(i, 0) + 1
    base = min(blocksize, symbols)
    entropy = 0
    for i in hist.values():
        p = i/float(blocksize)
        # If blocksize < 256, the number of possible byte values is restricted.
        # In that case, we adjust the log base to make sure we get a value
        # between 0 and 1.
        entropy += (p * math.log(p, base))
    return -entropy

os.chdir(sys.path[0])
bytesPerLine = 32
dataFileName = "conti.exe"
fileData = np.fromfile("../data/" + dataFileName, dtype="uint8")
paddedFileData = np.pad(fileData, (0, bytesPerLine - (np.size(fileData) % bytesPerLine)))
data = np.reshape(paddedFileData, (bytesPerLine, np.size(paddedFileData) // bytesPerLine))
#data = np.transpose(data)
#data = np.expand_dims(data, axis=2)
#imageData = np.tile(data, (1, 1, 3))
imageData = [[() for j in range()] for i in range()]

cv2.imwrite("../images/" + dataFileName + ".png", imageData)

