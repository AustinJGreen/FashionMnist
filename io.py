import PIL
import numpy as np

def readTrainData(filename):
    f = open(filename,'r')
    fileData = f.readlines() # Consume header line
    f.close()
    trainIds = np.zeros((60000,1),dtype=int)
    trainLabels = np.zeros((60000,1),dtype=int)
    trainImages = np.zeros((60000,28,28,1),dtype=float)
    for imageIndex in range(60000):
        # Extract image data
        imageData = fileData[imageIndex + 1].split(',')

        # Extract id and label
        trainIds[imageIndex] = int(imageData[0])
        trainLabels[imageIndex] = int(imageData[1])

        # Extract pixels
        # Pixels are stored in X = r * 28 + c
        # Pixels are valued [0, 255]
        imageMatrix = np.zeros((28,28,1), dtype=float)
        for x in range(784):
            dataIndex = x + 2;
            pixelValue = int(imageData[dataIndex])

            row = int(x / 28)
            column = int(x % 28)
            imageMatrix[column,row,0] = pixelValue / 255.0

        trainImages[imageIndex] = imageMatrix
        print("Loaded (%s/%s) train images." % (imageIndex, 60000))
    return trainIds, trainLabels, trainImages

def readTestData(filename):
    f = open(filename,'r')
    fileData = f.readlines() # Consume header line
    f.close()
    testIds = np.zeros((10000,1),dtype=int)
    testImages = np.zeros((10000,28,28,1),dtype=float)
    for imageIndex in range(10000):
        # Extract image data
        imageData = fileData[imageIndex + 1].split(',')

        # Extract id and label
        testIds[imageIndex] = int(imageData[0])

        # Extract pixels
        # Pixels are stored in X = r * 28 + c
        # Pixels are valued [0, 255]
        imageMatrix = np.zeros((28,28,1), dtype=float)
        for x in range(784):
            dataIndex = x + 1;
            pixelValue = int(imageData[dataIndex])

            row = int(x / 28)
            column = int(x % 28)
            imageMatrix[column,row,0] = pixelValue / 255.0

        testImages[imageIndex] = imageMatrix
        print("Loaded (%s/%s) test images." % (imageIndex, 10000))
    return testIds, testImages

def saveImages(directory, images):
    count = images.shape[0]
    for i in range(count):
        image = images[i]
        saveImage(format("%s/image%s.png" % (directory, i)), image)

# Saves a grayscale image [0, 1] to a file
def saveImage(filename, image):
    try:
        width = image.shape[0]
        height = image.shape[1]
        img = PIL.Image.new('RGB', (width, height), color=(0, 0, 0))
        for x in range(width):
            for y in range(height):
                grayScaleValue = image[x, y, 0] * 255
                r = grayScaleValue
                g = grayScaleValue
                b = grayScaleValue
                img.putpixel((x, y), (r, g, b))
        img.save(filename)
        img.close()
    except:
        print("Failed to save grayscale image to %s" % filename)

def generateClassificationFile(testIds,testLabels):
    f = open('./Data/prediction.csv', 'w')
    f.write("Id,label\n")
    for i in range(len(testLabels)):
        f.write(format("%s,%s\n" % (testIds[i][0], testLabels[i])))
    f.close()