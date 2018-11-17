import trainer
import fileutils
import processing
import subprocess
import os
import atexit
import time

def train():

    #TODO: Add resume to training
    #TODO: Try https://github.com/maxim5/hyper-engine

    # Load Training Data
    print("Loading training data...", end="", flush=True)
    _, trainLabelsRaw, trainImagesRaw = fileutils.readTrainDataRaw('./Data/train.csv')
    print("done.")

    # Normalize
    print("Normalizing data...", end="", flush=True)
    trainImages = processing.normalizeImages(trainImagesRaw)
    trainLabels = processing.convertLabels(trainLabelsRaw)
    print("done.")

    # Shuffle data very well
    print("Shuffling training data...", end="", flush=True)
    trainImages, trainLabels = processing.shuffle(trainImages, trainLabels)
    print("done.")

    print("Generating validation set...", end="", flush=True)
    validationSetSize = int(0.15 * trainImages.shape[0])
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

def eval():

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
    testLabels = trainer.evaluate(testImages)

    print("Generating classification CSV...", end="", flush=True)
    fileutils.generateClassificationFile(testIds,testLabels)
    print("done.")

def onExit():
    print("Closing objects....", end="", flush=True)
    trainer.closeTensorboard()
    print("done.")

if __name__ == "__main__":
    atexit.register(onExit)

    #eval()
    train()