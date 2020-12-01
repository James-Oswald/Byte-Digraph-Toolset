import os
import sys
os.chdir(sys.path[0])
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  #disable Tensorflow warning messages

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def naiveTest(bdms, lbls, numTest, classes):
    num_classes = len(classes)
    input_shape = (256, 256, 1)
    # the data, split between train and test sets
    #(trainingImages, trainingLabels), (testingImages, testingLabels) = keras.datasets.mnist.load_data()
    #datafile = np.load("../data/procTrainingData/realData.npz")
    
    x_data = bdms#datafile["data"]
    y_data = np.array(lbls) #datafile["labels"]
    x_data, y_data = unison_shuffled_copies(x_data, y_data)
    trainingImages = x_data[:numTest, :, :]
    trainingLabels = y_data[:numTest]
    testingImages = x_data[numTest:, :, :]
    testingLabels = y_data[numTest:]

    #encode lables 
    le = LabelEncoder()
    le.fit(trainingLabels)
    trainingLabels = le.transform(trainingLabels)
    le.fit(testingLabels)
    testingLabels = le.transform(testingLabels)

    # Make sure images have shape (256, 256, 1)
    trainingImages = np.expand_dims(trainingImages, -1)
    testingImages = np.expand_dims(testingImages, -1)
    #print("trainingImages shape:", trainingImages.shape)
    #print(trainingImages.shape[0], "train samples")
    #print(testingImages.shape[0], "test samples")

    # convert class vectors to binary class matrices
    trainingLabels = keras.utils.to_categorical(trainingLabels, num_classes)
    testingLabels = keras.utils.to_categorical(testingLabels, num_classes)

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Flatten(),
            layers.Dense(16, activation="sigmoid"),
            layers.Dense(16, activation="sigmoid"),
            layers.Dense(num_classes, activation="sigmoid"),
        ]
    )
    #model.summary()

    batch_size = 128
    epochs = 20
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(trainingImages, trainingLabels, batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=False)
    #model.save("../models/BBDM-Naive-Model.hdf5")

    score = model.evaluate(testingImages, testingLabels, verbose=0)
    #print("Test loss:", score[0])
    #print("Test accuracy:", score[1])
    return score[1]