import math

import scipy.ndimage
import numpy as np
import numpy.ma as ma


def reshape_images(raw_images):
    # Reshape 1d row-order one_hot_labels to 1 layer of 3d image for grayscale
    image_count = raw_images.shape[0]
    flattened_length = raw_images.shape[1]
    image2d_size = int(math.sqrt(flattened_length))

    # Use Fortran order to reshape, so reshape uses correct order from flattened array
    reshaped_images = np.reshape(raw_images, (image_count, image2d_size, image2d_size), order='F')
    return reshaped_images


def resize_images(reshaped_images, size):
    image_count = reshaped_images.shape[0]
    cur_size = reshaped_images.shape[1]
    scale_factor = size / cur_size
    resized_images = np.zeros((image_count, size, size))
    for i in range(image_count):
        resized_images[i] = scipy.ndimage.zoom(reshaped_images[i], scale_factor, order=3)

    return resized_images


def normalize_images(raw_images):

    # Divide by 255 to put range between [0, 1]
    normalized_images = raw_images / 255.0

    # Reshape 1d row-order one_hot_labels to 1 layer of 3d image for grayscale
    image_count = raw_images.shape[0]
    size = raw_images.shape[1]

    # Use Fortran order to reshape, so reshape uses correct order from flattened array
    normalized_images = np.reshape(normalized_images, (image_count, size, size, 1), order='F')

    return normalized_images


def convert_labels(raw_labels, categories=10,single=None):
    # Use label smoothing
    # http://www.deeplearningbook.org/contents/regularization.html
    # Section 7.5.1

    if single is not None:

        label_count = raw_labels.shape[0]
        soft_labels = np.full((label_count, 2), fill_value=0, dtype=int)
        for label in range(label_count):
            if raw_labels[label] == single:
                soft_labels[label, 0] = 1
            else:
                soft_labels[label, 1] = 1

        return soft_labels
    else:
        epsilon = 1e-10
        fill_value = epsilon / (categories - 1)
        hot_value = 1 - epsilon
        label_count = raw_labels.shape[0]

        soft_labels = np.full((label_count, 10), fill_value=fill_value, dtype=float)

        for label in range(label_count):
            soft_labels[label, raw_labels[label]] = hot_value

        return soft_labels


def shift_horizontal(arr, num):
    result = np.zeros(arr.shape)
    if num > 0:
        result[:, num:] = arr[:, :-num]
    elif num < 0:
        result[:, :num] = arr[:, -num:]
    else:
        result = arr
    return result


def shift_vertical(arr, num):
    result = np.zeros(arr.shape)
    if num > 0:
        result[:, :, num:] = arr[:, :, :-num]
    elif num < 0:
        result[:, :, :num] = arr[:, :, -num:]
    else:
        result = arr
    return result


def remove_random_part(images, kernel_size):
    width = images.shape[1]
    height = images.shape[2]

    x_start = np.random.randint(0, width - kernel_size)
    x_end = x_start + kernel_size

    y_start = np.random.randint(0, height - kernel_size)
    y_end = y_start + kernel_size

    images[:, x_start:x_end, y_start:y_end, :] = 0
    return images


def add_contrast(arr, min, max):
    result = np.ceil(arr) - 1  # Round up then flip 0 and 1's to set mask to true for all non-zero values
    mask = ma.masked_array(arr, mask=result, fill_value=0)  # Create mask

    # Create random matrix with values between min and max and same shape as arr
    rand_array = min + (np.random.rand(arr.shape[1], arr.shape[2], arr.shape[3]) * (max - min))

    # Add random contrasts to masked values
    result = mask + rand_array

    # Get result and clip values between 0 and 1 for any overflow from random
    result_data = ma.getdata(result)
    clipped = np.clip(result_data, 0, 1)
    return clipped


def augment_images(images, labels):

    # Flip images horizontally
    lr_flipped_images = np.fliplr(images)

    # Shift every image left,right,up,down, fill with black (0)
    shifted_right = shift_horizontal(images, 2)
    shifted_left = shift_horizontal(images, -2)
    shifted_up = shift_vertical(images, -2)
    shifted_down = shift_vertical(images, 2)

    # Add or subtract contrast from all the images
    contrasted = add_contrast(images, -0.02, 0.02)

    # Remove random part of image to help learn features
    # removed = remove_random_part(images, 4)

    # Add shifted images
    aug_images = np.concatenate(
        (images, lr_flipped_images, contrasted, shifted_right, shifted_left, shifted_up, shifted_down), axis=0)

    # Calculate amount of augmentation done so we can repeat/tile the labels
    repeats = int(aug_images.shape[0] / images.shape[0])

    aug_labels = np.tile(labels, (repeats, 1))

    return aug_images, aug_labels


def shuffle(images, labels, shuffles=10):
    if shuffles <= 0:
        return

    size = images.shape[0]
    permutation = np.random.permutation(size)
    shuffled_images = images[permutation]
    shuffled_labels = labels[permutation]
    for i in range(shuffles - 1):
        permutation = np.random.permutation(size)
        shuffled_images = shuffled_images[permutation]
        shuffled_labels = shuffled_labels[permutation]

    return shuffled_images, shuffled_labels
