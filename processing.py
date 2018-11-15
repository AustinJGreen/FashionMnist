import math
import numpy as np
import keras

def normalizeImages(rawImages):

    # Divide by 255 to put range between [0, 1]
    normalizedImages = rawImages / 255.0

    # Reshape 1d row-order matricies to 1 layer of 3d image for grayscale
    imageCount = normalizedImages.shape[0]
    flattenedLength = normalizedImages.shape[1]
    image2dSize = int(math.sqrt(flattenedLength))

    # Use Fortran order to reshape, so reshape uses correct order from flattened array
    normalizedImages = np.reshape(normalizedImages, (imageCount,image2dSize,image2dSize,1), order='F')
    return normalizedImages

def convertLabels(rawLabels):
    oneHotLabels = keras.utils.to_categorical(rawLabels, num_classes=10)
    return oneHotLabels

def augmentImages(images, labels):

    # Flip images
    flippedImages = np.fliplr(images)

    # Concatenate to images
    augImages = np.concatenate((images, flippedImages), axis=0)

    # Add labels
    augLabels = np.concatenate((labels, labels), axis=0)

    return augImages, augLabels
