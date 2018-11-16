from keras.models import Sequential
from keras.layers import InputLayer, MaxPooling2D, Conv2D, LeakyReLU, BatchNormalization, Dense, Softmax, Flatten, Dropout
from keras.optimizers import SGD, Adam
import numpy as np
import keras

def buildNN():

    weightInitializer = keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)

    net = Sequential()
    net.add(InputLayer((28, 28, 1)))

    net.add(Conv2D(256, kernel_size=5,kernel_initializer=weightInitializer))
    net.add(LeakyReLU(alpha=0.2))
    net.add(MaxPooling2D(pool_size=(2, 2)))

    net.add(Conv2D(128, kernel_size=5,kernel_initializer=weightInitializer))
    net.add(LeakyReLU(alpha=0.2))
    net.add(MaxPooling2D(pool_size=(2, 2)))

    net.add(Flatten())
    net.add(Dense(64, activation=None, kernel_initializer=weightInitializer))
    net.add(LeakyReLU(alpha=0.2))
    net.add(Dense(10, activation='softmax'))

    return net

def train(trainLabels, trainImages, validationSet):

    # Create network and configure optimizer
    net = buildNN()
    optimizer = SGD(lr=0.001)
    net.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Create callback for automatically saving best model based on highest validation accuracy
    checkpointCallback = keras.callbacks.ModelCheckpoint('model.h5', monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

    # Train network
    net.fit(trainImages,trainLabels,batch_size=100,epochs=200,verbose=2,shuffle=True,validation_data=validationSet,callbacks=[checkpointCallback])
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
