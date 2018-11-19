import math
import numpy as np
import numpy.ma as ma
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

def shift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result = arr
    return result

def addContrast(arr, minval, maxval):
    mask = np.putmask(arr, arr >= 0, 1)

def augmentImages(images, labels):

    # Flip images
    flippedImages = np.fliplr(images)

    # Concatenate to images
    augImages = np.concatenate((images, flippedImages), axis=0)

    # Shift every image left or right, fill with black (0)
    shiftedRight = shift(augImages, 2, cval=0)
    shiftedLeft = shift(augImages, -2, cval=0)

    # Add shifted images
    augImages = np.concatenate((augImages, shiftedLeft, shiftedRight), axis=0)

    # Add labels
    augLabels = np.concatenate((labels, labels, labels, labels), axis=0)
    return augImages, augLabels

def getDataGen():
    return keras.preprocessing.image.ImageDataGenerator(featurewise_center=False, samplewise_center=False,
                                                 featurewise_std_normalization=False,
                                                 samplewise_std_normalization=False, zca_whitening=False,
                                                 zca_epsilon=1e-06, rotation_range=0, width_shift_range=2.0,
                                                 height_shift_range=2.0, brightness_range=None, shear_range=0.0,
                                                 zoom_range=0.0, channel_shift_range=0.0, fill_mode='nearest', cval=0.0,
                                                 horizontal_flip=False, vertical_flip=False, rescale=None,
                                                 preprocessing_function=None, data_format=None)

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