import PIL
import numpy as np
import multiprocessing

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
                grayScaleValue = int(image[x, y, 0] * 255)
                r = grayScaleValue
                g = grayScaleValue
                b = grayScaleValue
                img.putpixel((x, y), (r, g, b))
        img.save(filename)
        img.close()
    except Exception as e:
        print("Failed to save grayscale image to %s" % filename)
        print(e)

def generateClassificationFile(testIds,testLabels):
    f = open('./prediction.csv', 'w')
    f.write("Id,label\n")
    for i in range(len(testLabels)):
        f.write(format("%s,%s\n" % (testIds[i], testLabels[i])))
    f.close()