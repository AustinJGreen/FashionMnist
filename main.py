import trainer
import fileutils
import processing
import numpy as np
import tests

def train():
    # TODO: Auto submit csv file to kaggle with api
    # TODO: Separate runs in folder directories and show all on tensorboard

    # Load Training Data
    print("Loading training data...", end="", flush=True)
    _, yTrain, xTrain = fileutils.readTrainDataRaw('./Data/train.csv')
    print("done.")

    # Normalize
    print("Normalizing data...", end="", flush=True)
    trainImages = processing.normalizeImages(xTrain)
    trainLabels = processing.convertLabels(yTrain)
    print("done.")

    # Shuffle data very well
    print("Shuffling training data...", end="", flush=True)
    trainImages, trainLabels = processing.shuffle(trainImages, trainLabels)
    print("done.")

    # Test distribution
    labelSums = np.sum(trainLabels, axis=0)


    print("Generating validation set...", end="", flush=True)
    validationSetSize = int(0 * trainImages.shape[0])
    validationSet = None
    if validationSetSize > 0:
        validationSet = (trainImages[-validationSetSize:], trainLabels[-validationSetSize:])
        trainImages = trainImages[:-validationSetSize]
        trainLabels = trainLabels[:-validationSetSize]
    print("done.")

    # Run Test
    print("Generating augmented training set...", end="", flush=True)
    augTrainImages, augTrainLabels = processing.augmentImages(trainImages, trainLabels)
    print("done.")

    # Shuffle data very well
    print("Shuffling augmented data...", end="", flush=True)
    augTrainImages, augTrainLabels = processing.shuffle(augTrainImages, augTrainLabels)
    print("done.")

    trainer.train(augTrainLabels, augTrainImages, validationSet)

def eval(name):

    # Load Test Data
    print("Loading test data...", end="", flush=True)
    testIds, testImagesRaw = fileutils.readTestData('./Data/test.csv')
    print("done.")

    # Normalize
    print("Normalizing data...", end="", flush=True)
    testImages = processing.normalizeImages(testImagesRaw)
    print("done.")

    # Load best
    print("Evaluating test set...")
    testLabels = trainer.evaluate(testImages, name)

    print("Generating classification CSV...", end="", flush=True)
    fileutils.generateClassificationFile(testIds,testLabels)
    print("done.")

def checkPaths():
    fileutils.checkPath("Data")
    fileutils.checkPath("Images")
    fileutils.checkPath("Tensorboard")

if __name__ == "__main__":
    checkPaths()

    eval('latest.h5')
    #train()