from keras.models import Sequential
from keras.layers import InputLayer, MaxPooling2D, Conv2D, LeakyReLU, BatchNormalization, Dense, Softmax, Flatten, Dropout, SpatialDropout2D
from keras.optimizers import SGD, Adam
from keras.models import load_model
import numpy as np
import keras

def buildNN():

    kernelInit = keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)

    # NOTE: Don't use input layer, otherwise the model cannot be loaded from an h5 file due to a bug in keras

    net = Sequential()
    net.add(Conv2D(256, kernel_size=5,kernel_initializer=kernelInit, input_shape=(28, 28, 1)))
    net.add(BatchNormalization(momentum=0.8))
    net.add(LeakyReLU(alpha=0.2))
    net.add(MaxPooling2D(pool_size=(2, 2)))
    net.add(SpatialDropout2D(0.5))

    net.add(Conv2D(128, kernel_size=5,kernel_initializer=kernelInit))
    net.add(BatchNormalization(momentum=0.8))
    net.add(LeakyReLU(alpha=0.2))
    net.add(MaxPooling2D(pool_size=(2, 2)))
    net.add(SpatialDropout2D(0.5))

    net.add(Flatten())
    net.add(Dense(64, activation=None, kernel_initializer=kernelInit))
    net.add(LeakyReLU(alpha=0.2))
    net.add(Dense(10, activation='softmax'))

    return net

def train(trainLabels, trainImages, validationSet):

    # Create network and configure optimizer
    net = buildNN()
    optimizer = SGD(lr=0.01)
    net.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Create callback for automatically saving best model based on highest validation accuracy
    checkpointCallback = keras.callbacks.ModelCheckpoint('best.h5', monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

    # Create callback for tensorboard
    tbCallback = keras.callbacks.TensorBoard(log_dir='./Tensorboard', histogram_freq=0, batch_size=500, write_graph=True, write_grads=True,
                                write_images=True, embeddings_freq=0, embeddings_layer_names=None,
                                embeddings_metadata=None, embeddings_data=None)

    # Train network and save best model along the way
    net.fit(trainImages,trainLabels,batch_size=500,epochs=1000,verbose=2,shuffle=True,validation_data=validationSet,callbacks=[checkpointCallback, tbCallback])

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
