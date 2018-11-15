import numpy as np
import trainer
import utils

def main():
    #TODO: Load data
    #TODO: Augment data
    #TODO: https://keras.io/initializers/
    #TODO: Tensorboard
    #TODO: BatchNorm
    #- https://snow.dog/blog/data-augmentation-for-small-datasets

    trainIds, trainLabels, trainImages = utils.readTrainData('./Data/train.csv')
    utils.saveImages('./Images/Training', trainImages)

    testIds, testImages = io.readTestData('./Data/test.csv')

    #trainedNet = trainer.train(trainIds, trainLabels, trainImages)
    #testLabels = trainer.evaluate(trainedNet, testImages)

    #io.generateClassificationFile(testIds,testLabels)

if __name__ == "__main__":
    main();