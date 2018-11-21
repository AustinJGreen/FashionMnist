import math

import keras
import numpy as np
import numpy.ma as ma


def normalizeImages(rawImages):
    # Divide by 255 to put range between [0, 1]
    normalizedImages = rawImages / 255.0

    # Reshape 1d row-order matricies to 1 layer of 3d image for grayscale
    imageCount = normalizedImages.shape[0]
    flattenedLength = normalizedImages.shape[1]
    image2dSize = int(math.sqrt(flattenedLength))

    # Use Fortran order to reshape, so reshape uses correct order from flattened array
    normalizedImages = np.reshape(normalizedImages, (imageCount, image2dSize, image2dSize, 1), order='F')
    return normalizedImages


def convertLabels(rawLabels):
    oneHotLabels = keras.utils.to_categorical(rawLabels, num_classes=10)
    return oneHotLabels


def shiftHorizontal(arr, num):
    result = np.zeros(arr.shape)
    if num > 0:
        result[:, num:] = arr[:, :-num]
    elif num < 0:
        result[:, :num] = arr[:, -num:]
    else:
        result = arr
    return result


def shiftVertical(arr, num):
    result = np.zeros(arr.shape)
    if num > 0:
        result[:, :, num:] = arr[:, :, :-num]
    elif num < 0:
        result[:, :, :num] = arr[:, :, -num:]
    else:
        result = arr
    return result


def addContrast(arr, min, max):
    result = np.ceil(arr) - 1  # Round up then flip 0 and 1's to set mask to true for all non-zero values
    mask = ma.masked_array(arr, mask=result, fill_value=0)  # Create mask

    # Create random matrix with values between min and max and same shape as arr
    randArray = min + (np.random.rand(arr.shape[1], arr.shape[2], arr.shape[3]) * (max - min))

    # Add random contrasts to masked values
    result = mask + randArray

    # Get result and clip values between 0 and 1 for any overflow from random
    resultData = ma.getdata(result)
    clipped = np.clip(resultData, 0, 1)
    return clipped


def augmentImages(images, labels):
    # Flip images
    flippedImages = np.fliplr(images)

    # Shift every image left,right,up,down, fill with black (0)
    shiftedRight = shiftHorizontal(images, 2)
    shiftedLeft = shiftHorizontal(images, -2)

    # Add or subtract contrast from all the images
    contrasted = addContrast(images, -0.02, 0.02)

    # Add shifted images
    augImages = np.concatenate((images, flippedImages, shiftedLeft, shiftedRight, contrasted), axis=0)

    # Calculate amount of augmentation done so we can repeat the labels
    repeats = int(augImages.shape[0] / images.shape[0])

    augLabels = np.tile(labels, (repeats, 1))
    return augImages, augLabels


def shuffle(images, labels):
    size = images.shape[0]
    permutation = np.random.permutation(size)
    shuffledImages = images[permutation]
    shuffledLabels = labels[permutation]
    for i in range(9):
        permutation = np.random.permutation(size)
        shuffledImages = shuffledImages[permutation]
        shuffledLabels = shuffledLabels[permutation]

    return shuffledImages, shuffledLabels
