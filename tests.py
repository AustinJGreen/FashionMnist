import numpy as np
import fileutils
from keras.models import Sequential
import customlayers

def testContrastLayer(testImages):

    net = Sequential()
    net.add(customlayers.UniformNoise(minval=-0.02,maxval=0.02))

    output = net.predict(testImages)

    saveImages("./Images", output, 100)

    return

def saveImages(directory, imageSet, count):
    imageSetCount = imageSet.shape[0]
    for i in range(count):
        randomIndex = np.random.randint(0, imageSetCount)
        image = imageSet[randomIndex]
        fileutils.saveImage(format("%s/image%s.png" % (directory, randomIndex)), image)