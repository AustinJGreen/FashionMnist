from keras.models import Sequential
from keras.layers import InputLayer, MaxPooling2D, Conv2D, LeakyReLU, BatchNormalization, Dense, Softmax, Flatten, Dropout, SpatialDropout2D, GaussianNoise
from keras.optimizers import SGD, Adam
from keras.models import load_model
import numpy as np
import os
from subprocess import Popen
import keras

def buildNN():

    kernelInit = keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=None)

    # NOTE: Don't use input layer, otherwise the model cannot be loaded from an h5 file due to a bug in keras
    net = Sequential()
    net.add(GaussianNoise(stddev=0.1, input_shape=(28, 28, 1)))
    net.add(Conv2D(256, kernel_size=5,kernel_initializer=kernelInit))
    net.add(BatchNormalization(momentum=0.8))
    net.add(LeakyReLU(alpha=0.2))
    net.add(MaxPooling2D(pool_size=(2, 2)))
    net.add(SpatialDropout2D(0.25))

    net.add(Conv2D(128, kernel_size=3,kernel_initializer=kernelInit))
    net.add(BatchNormalization(momentum=0.8))
    net.add(LeakyReLU(alpha=0.2))
    net.add(MaxPooling2D(pool_size=(2, 2)))
    net.add(SpatialDropout2D(0.25))

    net.add(Flatten())
    net.add(Dense(64, activation=None, kernel_initializer=kernelInit))
    net.add(LeakyReLU(alpha=0.2))
    net.add(Dense(10, activation='softmax'))

    return net

def startTensorboard():
    # Delete old tensorflow data
    paths = os.listdir('./Tensorboard')
    for path in paths:
        fullpath = './Tensorboard/%s' % path
        if os.path.isfile(fullpath):
            os.remove(fullpath)

    # Start new tensorboard instance
    currentPath = os.getcwd()
    logDir = '%s\\Tensorboard' % currentPath
    Popen(['tensorboard', '--logdir=%s' % logDir], shell=True)

def closeTensorboard():
    # Kill any tensorboard tasks
    Popen(['taskkill', '/IM', 'tensorboard.exe', '/F'])

def train(trainLabels, trainImages, validationSet):

    # Create network and configure optimizer
    net = buildNN()
    optimizer = SGD(lr=0.01)
    net.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Create callback for automatically saving best model based on highest validation accuracy
    checkBestCallback = keras.callbacks.ModelCheckpoint('best.h5', monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

    # Create callback for automatically saving lastest model so training can be resumed. Saves every 10 epochs
    checkLatestCallback = keras.callbacks.ModelCheckpoint('latest.h5', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=10)

    # Create callback to automatically lower the learning rate when the accuracy pleateaus
    autoReduceLRCallback = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.1, verbose=1, mode='auto', cooldown=100)

    # Create callback for tensorboard
    tbCallback = keras.callbacks.TensorBoard(log_dir='./Tensorboard', histogram_freq=0, batch_size=250, write_graph=True, write_grads=True,
                                write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                embeddings_metadata=None, embeddings_data=None)

    # Create list of all callbacks
    callbackList = [ checkBestCallback, checkLatestCallback, autoReduceLRCallback, tbCallback ]

    # Start tensorboard
    startTensorboard()

    # Train network and save best model along the way
    net.fit(trainImages,trainLabels,batch_size=250,epochs=2000,verbose=2,shuffle=True,validation_data=validationSet,callbacks=callbackList)

def evaluate(testImages):
    network = load_model('best.h5')

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
