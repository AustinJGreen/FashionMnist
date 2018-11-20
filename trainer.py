from keras.models import Sequential
from keras.layers import Conv2D, LeakyReLU, BatchNormalization, Dense, Softmax, Flatten, SpatialDropout2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.models import load_model
import psutil
import os
from subprocess import Popen
import keras
import fileutils

def buildNetwork():

    #kernelInit = keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=None)
    #kernelRegulizer = keras.regularizers.l2(0.001)
    #br = keras.regularizers.l2(0.001)

    net = Sequential()
    net.add(Conv2D(64, kernel_size=5, input_shape=(28, 28, 1)))
    net.add(BatchNormalization(momentum=0.8))
    net.add(LeakyReLU(alpha=0.2))
    net.add(MaxPooling2D(pool_size=(2, 2)))
    net.add(SpatialDropout2D(0.25))

    net.add(Conv2D(32, kernel_size=3))
    net.add(BatchNormalization(momentum=0.8))
    net.add(LeakyReLU(alpha=0.2))
    net.add(MaxPooling2D(pool_size=(2, 2)))
    net.add(SpatialDropout2D(0.25))

    net.add(Flatten())
    net.add(Dense(128, activation=None))
    net.add(LeakyReLU(alpha=0.2))
    net.add(BatchNormalization(momentum=0.8))
    net.add(Dense(10, activation='softmax'))

    return net

def startTensorboard():
    # Check if tensorboard is already running
    for p in psutil.process_iter():
        if p.name() == "tensorboard.exe":
            return

    # Start new tensorboard instance
    currentPath = os.getcwd()
    logDir = '%s\\Tensorboard' % currentPath
    Popen(['tensorboard', '--logdir=%s' % logDir], shell=True)

def deleteTensorboardData(runName):
    currentPath = os.getcwd()
    baseDir = '%s\\Tensorboard\\%s\\' % (currentPath, runName)
    paths = os.listdir(baseDir)
    for path in paths:
        fullpath = '%s\\%s' % (baseDir, path)
        if os.path.isfile(fullpath):
            os.remove(fullpath)

def trainNew(runName, trainLabels, trainImages, validationSet):

    # Create run in tensoboard directory
    tbDir = "./Tensorboard/%s" % runName
    os.makedirs(tbDir)

    # Create Models directory
    modelsDir = './Runs/%s/Models' % runName
    os.makedirs(modelsDir)

    # Create network and configure optimizer
    net = buildNetwork()

    # Save network architecture
    yamlStr = net.to_yaml()
    fileutils.saveText('./Runs/%s/architecture.yaml' % runName, yamlStr)

    # Set batch size
    batchSize = 32

    # Compile new network with optimizer
    optimizer = Adam(lr=0.0001)
    net.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Create callback for automatically saving best model based on highest validation accuracy
    checkBestCallback = keras.callbacks.ModelCheckpoint('%s/best.h5' % modelsDir, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

    # Create callback for automatically saving lastest model so training can be resumed. Saves every epoch
    checkLatestCallback = keras.callbacks.ModelCheckpoint('%s/latest.h5' % modelsDir, verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

    # Create callback for tensorboard
    tbCallback = keras.callbacks.TensorBoard(log_dir=tbDir, batch_size=batchSize, write_graph=False, write_grads=True)

    # Create list of all callbacks
    callbackList = [ checkLatestCallback, tbCallback ]
    if validationSet is not None:
        callbackList = callbackList.append(checkBestCallback)

    # Start tensorboard
    startTensorboard()

    # Train network and save best model along the way
    net.fit(trainImages,trainLabels,batch_size=batchSize,epochs=150,verbose=2,shuffle=True,validation_data=validationSet,callbacks=callbackList)

def trainExisting(runName, modelPath, trainLabels, trainImages, validationSet):

    # Get tensoboard directory
    tbDir = "./Tensorboard/%s" % runName

    # Delete old data because epochs are being reset to 0
    deleteTensorboardData(runName)

    # Get Models directory
    modelsDir = './Runs/%s/Models' % runName

    # Set batch size
    batchSize = 32

    # Create callback for automatically saving best model based on highest validation accuracy
    checkBestCallback = keras.callbacks.ModelCheckpoint('%s/best.h5' % modelsDir, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

    # Create callback for automatically saving lastest model so training can be resumed. Saves every epoch
    checkLatestCallback = keras.callbacks.ModelCheckpoint('%s/latest.h5' % modelsDir, verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

    # Create callback for tensorboard
    tbCallback = keras.callbacks.TensorBoard(log_dir=tbDir, batch_size=batchSize, write_graph=False, write_grads=True)

    # Create list of all callbacks
    callbackList = [ checkLatestCallback, tbCallback ]
    if validationSet is not None:
        callbackList = callbackList.append(checkBestCallback)

    # Start tensorboard
    startTensorboard()

    # Load network from file
    net = load_model(modelPath)

    # Train network and save best model along the way
    net.fit(trainImages,trainLabels,batch_size=batchSize,epochs=150,verbose=2,shuffle=True,validation_data=validationSet,callbacks=callbackList)

def evaluate(testImages, modelPath):

    # Load network fron h5 format
    net = load_model(modelPath)

    # Feed test images into network and get predictions
    onehotPredictions = net.predict(testImages)

    # Convert labels back from one-hot to single integer
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
