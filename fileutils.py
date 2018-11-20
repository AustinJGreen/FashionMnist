import PIL
import numpy as np
import os

def checkPath(name):
    baseDir = os.getcwd()
    localDir = "%s\\%s\\" % (baseDir, name)
    if not os.path.exists(localDir):
        os.makedirs(localDir)
        return True

    return False

def readCsvData(filename):
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

def readTrainDataRaw(filename):
    csvData = readCsvData(filename)
    trainIds = csvData[:,0]
    trainLabels = csvData[:,1]
    trainImages = csvData[:,2:]
    return trainIds, trainLabels, trainImages

def readTestData(filename):
    csvData = readCsvData(filename)
    testIds = csvData[:, 0]
    testImages = csvData[:, 1:]
    return testIds, testImages

# Saves a grayscale image [0, 1] to a file
def saveImage(filename, image):
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

def generateClassificationFile(testIds, testLabels, runName):
    curDir = os.getcwd()
    with open('%s\\Runs\\%s\\prediction.csv' % (curDir, runName), 'w') as f:
        f.write("Id,label\n")
        for i in range(len(testLabels)):
            f.write(format("%s,%s\n" % (testIds[i], testLabels[i])))

def saveText(filename, text):
    f = open(filename, 'w')
    f.write(text)
    f.close()
