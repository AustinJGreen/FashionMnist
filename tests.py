import numpy as np
import fileutils

def saveImages(directory, imageSet, count):
    imageSetCount = imageSet.shape[0]
    for i in range(count):
        randomIndex = np.random.randint(0, imageSetCount)
        image = imageSet[randomIndex]
        fileutils.saveImage(format("%s/image%s.png" % (directory, randomIndex)), image)