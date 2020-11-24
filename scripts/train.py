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

num_classes = 3
input_shape = (256, 256, 1)
# the data, split between train and test sets
#(trainingImages, trainingLabels), (testingImages, testingLabels) = keras.datasets.mnist.load_data()
datafile = np.load("../data/procTrainingData/data.npz")
x_data = datafile["data"]
y_data = datafile["labels"]
x_data, y_data = unison_shuffled_copies(x_data, y_data)
trainingImages = x_data[:2500, :, :]
trainingLabels = y_data[:2500]
testingImages = x_data[2500:, :, :]
testingLabels = y_data[2500:]

#encode lables 
le = LabelEncoder()
le.fit(trainingLabels)
trainingLabels = le.transform(trainingLabels)
le.fit(testingLabels)
testingLabels = le.transform(testingLabels)

# Make sure images have shape (256, 256, 1)
trainingImages = np.expand_dims(trainingImages, -1)
testingImages = np.expand_dims(testingImages, -1)
print("trainingImages shape:", trainingImages.shape)
print(trainingImages.shape[0], "train samples")
print(testingImages.shape[0], "test samples")

# convert class vectors to binary class matrices
trainingLabels = keras.utils.to_categorical(trainingLabels, num_classes)
testingLabels = keras.utils.to_categorical(testingLabels, num_classes)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)
model.summary()

batch_size = 128
epochs = 3
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(trainingImages, trainingLabels, batch_size=batch_size, epochs=epochs, validation_split=0.1)
model.save("../data/models/BBDMModel.hdf5")

score = model.evaluate(testingImages, testingLabels, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])