import PIL
import numpy as np
import os

def check_path(name):
    """
    Checks if the specified path exists, and if it doesn't, creates it
    :param name: The path to check or create
    :return: True if pre-existing; otherwise False
    """

    baseDir = os.getcwd()
    localDir = "%s\\%s\\" % (baseDir, name)
    if not os.path.exists(localDir):
        os.makedirs(localDir)
        return True

    return False


def read_csv_data(filename):
    """
    Reads raw CSV data from a file
    :param filename: CSV file
    :return: Data from the CSV file formatted as a matrix
    """

    f = open(filename, 'r')
    fileData = f.readlines()  # Consume header line
    f.close()

    header = fileData[0]
    columns = len(header.split(','))
    rows = len(fileData) - 1

    dataMatrix = np.zeros((rows, columns), dtype=int)
    for r in range(rows):
        currentRowData = fileData[r + 1].split(',')
        for c in range(columns):
            dataMatrix[r, c] = int(currentRowData[c])

    return dataMatrix


def read_train_data_raw(filename):
    """
    Reads raw training data from kaggle's CSV file
    :param filename: The CSV file
    :return: Tuple of training IDs, training Labels, and training Images
    """

    csvData = read_csv_data(filename)
    trainIds = csvData[:, 0]
    trainLabels = csvData[:, 1]
    trainImages = csvData[:, 2:]
    return trainIds, trainLabels, trainImages


def read_test_data(filename):
    """
    Reads raw test data from kaggle's CSV file
    :param filename: The CSV file
    :return: Tuple of test IDs and test Images
    """

    csvData = read_csv_data(filename)
    testIds = csvData[:, 0]
    testImages = csvData[:, 1:]
    return testIds, testImages


def save_image(filename, image):
    """
    Saves a [0, 1] grayscale image to a file
    :param filename: The file to save the image to
    :param image: The [0, 1] grayscale image to save
    """
    try:
        width = image.shape[0]
        height = image.shape[1]
        img = PIL.Image.new('RGB', (width, height), color=(0, 0, 0))
        for x in range(width):
            for y in range(height):
                grayValue = int(image[x, y, 0] * 255)
                img.putpixel((x, y), (grayValue, grayValue, grayValue))
        img.save(filename)
        img.close()
    except Exception as e:
        print("Failed to save grayscale image to %s" % filename)
        print(e)


def generate_classification(testIds, testLabels, runName):
    """
    Generates a CSV classification file for a network that can be submitted to kaggle
    :param testIds: The test ids to attach to each label
    :param testLabels: The labels corresponding to each test ID
    :param runName: The test IDs corresponding to each label
    """

    assert testIds.shape[0] == testLabels.shape[0], "Test IDs and Test Labels must have the same length"

    curDir = os.getcwd()
    with open('%s\\Runs\\%s\\prediction.csv' % (curDir, runName), 'w') as f:
        f.write("Id,label\n")
        for i in range(len(testLabels)):
            f.write(format("%s,%s\n" % (testIds[i], testLabels[i])))


def save_text(filename, text):
    f = open(filename, 'w')
    f.write(text)
    f.close()
