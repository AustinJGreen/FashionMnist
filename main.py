import numpy as np
import trainer

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

def drawImages(imageDataset):
    return

def generateClassificationFile(testIds,testLabels):
    f = open('./Data/prediction.csv', 'w')
    f.write("Id,label\n")
    for i in range(len(testLabels)):
        f.write(format("%s,%s\n" % (testIds[i][0], testLabels[i])))
    f.close()

def main():
    #TODO: Load data
    trainIds, trainLabels, trainImages = readTrainData('./Data/train.csv')
    testIds, testImages = readTestData('./Data/test.csv')

    trainedNet = trainer.train(trainIds, trainLabels, trainImages)
    testLabels = trainer.evaluate(trainedNet, testImages)

    generateClassificationFile(testIds,testLabels)

if __name__ == "__main__":
    main();