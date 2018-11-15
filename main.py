import numpy as np
import trainer
import fileutils
import processing
import tests
import time

def main():

    #TODO: https://keras.io/initializers/
    #TODO: Tensorboard
    #TODO: BatchNorm

    # Load Training Data
    print("Loading training data...", end="", flush=True)
    _, trainLabelsRaw, trainImagesRaw = fileutils.readTrainDataRaw('./Data/train.csv')
    print("done.")

    # Load Test Data
    print("Loading test data...", end="", flush=True)
    testIds, testImagesRaw = fileutils.readTestData('./Data/test.csv')
    print("done.")

    # Normalize
    print("Normalizing data...", end="", flush=True)
    trainImages = processing.normalizeImages(trainImagesRaw)
    testImages = processing.normalizeImages(testImagesRaw)
    trainLabels = processing.convertLabels(trainLabelsRaw)
    print("done.")

    print("Generating validation set...", end="", flush=True)
    validationSetSize = int(0.2 * trainImages.shape[0])
    validationSet = (trainImages[-validationSetSize:], trainLabels[-validationSetSize:])
    trainImages = trainImages[:-validationSetSize]
    trainLabels = trainLabels[:-validationSetSize]
    print("done.")

    # Run Test
    print("Generating augmented training set...", end="", flush=True)
    augTrainImages, augTrainLabels = processing.augmentImages(trainImages, trainLabels)
    print("done.")

    trainedNet = trainer.train(augTrainLabels, augTrainImages, validationSet)
    testLabels = trainer.evaluate(trainedNet, testImages)

    fileutils.generateClassificationFile(testIds,testLabels)

if __name__ == "__main__":
    main();