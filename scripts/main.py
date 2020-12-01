
from BDM import BBDM, PDBDM
from naiveTrain import naiveTest
from cnnTrain import cnnTest
from realData import getRealDataBDMs
import matplotlib.pyplot as plt

samplesPerClass = 150
classes = ["audio", "text", "exedata", "dlldata", "pngs", "jpgs", "random"]

bytesPerBDM = range(5000, 35000, 5000)
results = []

for i in range(len(bytesPerBDM)):
    bdms, lbls = getRealDataBDMs(BBDM, bytesPerBDM[i], samplesPerClass, classes)
    print("Finished Generating BBDMs for " + str(bytesPerBDM[i]))
    acc = cnnTest(bdms, lbls, int(len(classes) * samplesPerClass * 0.8), classes)
    results.append(acc)
    print("Finished Training for " + str(bytesPerBDM[i]) + " acc was " + str(acc))

plt.plot(bytesPerBDM, results)
plt.savefig("../results/naiveBBDMs8.png")
plt.show()