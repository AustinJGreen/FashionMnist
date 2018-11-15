from keras.models import Sequential
from keras.layers import InputLayer, MaxPooling2D, Conv2D, LeakyReLU, BatchNormalization, Dense, Softmax, Flatten, Dropout
from keras.optimizers import SGD, Adam
import numpy as np
import keras

def buildNN():

    net = Sequential()
    net.add(InputLayer((28, 28, 1)))

    net.add(Conv2D(128, kernel_size=3))
    net.add(LeakyReLU(alpha=0.2))
    net.add(MaxPooling2D(pool_size=(2, 2)))
    net.add(Dropout(0.1))

    net.add(Conv2D(64, kernel_size=5))
    net.add(LeakyReLU(alpha=0.2))
    net.add(MaxPooling2D(pool_size=(2, 2)))
    net.add(Dropout(0.1))

    net.add(Flatten())
    net.add(Dense(32, activation=None))
    net.add(LeakyReLU(alpha=0.2))
    net.add(Dense(10, activation='softmax'))

    return net

def train(trainIds, trainLabels, trainImages):
    net = buildNN()

    # Convert labels to categorical one-hot encoding
    onehotLabels = keras.utils.to_categorical(trainLabels, num_classes=10)

    #Train network
    optimizer = Adam(lr=0.0001, beta_1=0.5)
    net.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    net.fit(trainImages,onehotLabels,batch_size=100,epochs=100,verbose=1,shuffle=True,validation_split=0.15)
    return net

def evaluate(network, testImages):
    onehotPredictions = network.predict(testImages)

    testLabels = [0] * onehotPredictions.shape[0]
    for i in range(onehotPredictions.shape[0]):
        highestValue = -1
        highestIndex = -1
        for j in range(onehotPredictions.shape[1]):
            if (onehotPredictions[i][j] > highestValue):
                highestValue = onehotPredictions[i][j]
                highestIndex = j;
        testLabels[i] = highestIndex
    return testLabels
