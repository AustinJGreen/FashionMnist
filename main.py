import numpy as np
import trainer
import io

def main():
    #TODO: Load data
    trainIds, trainLabels, trainImages = io.readTrainData('./Data/train.csv')
    io.saveImages('./Images/Training', trainImages)

    #testIds, testImages = io.readTestData('./Data/test.csv')

    #trainedNet = trainer.train(trainIds, trainLabels, trainImages)
    #testLabels = trainer.evaluate(trainedNet, testImages)

    #io.generateClassificationFile(testIds,testLabels)

if __name__ == "__main__":
    main();