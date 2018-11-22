import os
from zipfile import ZipFile

import PIL
import numpy as np


def check_path(name):
    """
    Checks if the specified path exists, and if it doesn't, creates it
    :param name: The path to check or create
    :return: True if pre-existing; otherwise False
    """

    base_dir = os.getcwd()
    local_dir = "%s\\%s\\" % (base_dir, name)
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
        return True

    return False


def read_csv_data(filename):
    """
    Reads raw CSV data from a file
    :param filename: CSV file
    :return: Data from the CSV file formatted as a matrix
    """

    f = open(filename, 'r')
    file_data = f.readlines()  # Consume header line
    f.close()

    header = file_data[0]
    columns = len(header.split(','))
    rows = len(file_data) - 1

    data_matrix = np.zeros((rows, columns), dtype=int)
    for r in range(rows):
        current_row_data = file_data[r + 1].split(',')
        for c in range(columns):
            data_matrix[r, c] = int(current_row_data[c])

    return data_matrix


def read_train_data_raw(filename):
    """
    Reads raw training data from kaggle's CSV file
    :param filename: The CSV file
    :return: Tuple of training IDs, training Labels, and training Images
    """

    csv_data = read_csv_data(filename)
    train_ids = csv_data[:, 0]
    train_labels = csv_data[:, 1]
    train_images = csv_data[:, 2:]
    return train_ids, train_labels, train_images


def read_test_data(filename):
    """
    Reads raw test data from kaggle's CSV file
    :param filename: The CSV file
    :return: Tuple of test IDs and test Images
    """

    csv_data = read_csv_data(filename)
    test_ids = csv_data[:, 0]
    test_images = csv_data[:, 1:]
    return test_ids, test_images


def save_image(filename, image):
    """
    Saves a [0, 1] grayscale image to a file
    :param filename: The file to save the image to
    :param image: The [0, 1] grayscale image to save
    """
    try:
        width = image.shape[0]
        height = image.shape[1]
        with PIL.Image.new('RGB', (width, height), color=(0, 0, 0)) as image:
            for x in range(width):
                for y in range(height):
                    gray_value = int(image[x, y, 0] * 255)
                    image.putpixel((x, y), (gray_value, gray_value, gray_value))
            image.save(filename)
    except Exception as e:
        print("Failed to save grayscale image to %s" % filename)
        print(e)


def save_images(directory, image_set, count):
    """
    Saves a random batch of grayscale images to a directory
    :param directory: Directory to save the images to
    :param image_set: Image set to draw grayscale [0, 1] images from
    :param count: The amount of images to save
    """

    image_set_count = image_set.shape[0]
    for i in range(count):
        random_index = np.random.randint(0, image_set_count)
        image = image_set[random_index]
        save_image(format("%s/image%s.png" % (directory, random_index)), image)


def generate_classification(test_ids, test_labels, run_name):
    """
    Generates a CSV classification file for a network that can be submitted to kaggle
    :param test_ids: The test ids to attach to each label
    :param test_labels: The labels corresponding to each test ID
    :param run_name: The test IDs corresponding to each label
    """

    assert len(test_ids) == len(test_labels), "Test IDs and Test Labels must have the same length"

    cur_dir = os.getcwd()
    try:
        with open('%s\\Runs\\%s\\prediction.csv' % (cur_dir, run_name), 'w') as f:
            f.write("Id,label\n")
            for i in range(len(test_labels)):
                f.write(format("%s,%s\n" % (test_ids[i], test_labels[i])))
    except Exception as e:
        print("Failed to generate classification")
        print(e)


def save_current_code(filename):
    try:
        with ZipFile(filename, 'w') as codebase:

            # Loop through current directory and add all python files
            cur_dir = os.getcwd()
            for filename in os.listdir(cur_dir):
                if os.path.isfile(filename) and filename.endswith('.py'):
                    codebase.write(filename)
    except Exception as e:
        print('Failed to zip code')
        print(e)


def save_text(filename, text):
    try:
        with open(filename, 'w') as f:
            f.write(text)
    except Exception as e:
        print("Failed to save text to %s" % filename)
        print(e)


def read_text(filename):
    assert os.path.exists(filename), "Path does not exist, cannot read text."

    try:
        with open(filename, 'r') as f:
            return f.read()
    except Exception as e:
        print("Failed to save text to %s" % filename)
        print(e)

    return None
