import trainer
import fileutils
import processing
import numpy as np
import os
import tests

# TODO: Auto submit csv file to kaggle with api
# TODO: Separate runs in folder directories and show all on tensorboard

def loadTrainingData():
    # Load Training Data
    print("Reading training data...", end="", flush=True)
    _, yTrain, xTrain = fileutils.read_train_data_raw('./Data/train.csv')
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

    print("Generating validation set...", end="", flush=True)
    validationSetSize = int(0 * trainImages.shape[0])
    validationSet = None
    if validationSetSize > 0:
        validationSet = (trainImages[-validationSetSize:], trainLabels[-validationSetSize:])

        # TODO: Test distribution
        maxSums = np.sum(trainLabels, axis=0)

        trainImages = trainImages[:-validationSetSize]
        trainLabels = trainLabels[:-validationSetSize]

        labelSums = np.sum(trainLabels, axis=0)
    print("done.")

    # Run Test
    print("Generating augmented training set...", end="", flush=True)
    augTrainImages, augTrainLabels = processing.augmentImages(trainImages, trainLabels)
    print("done.")

    # Shuffle data very well
    print("Shuffling augmented data...", end="", flush=True)
    augTrainImages, augTrainLabels = processing.shuffle(augTrainImages, augTrainLabels)
    print("done.")

    return augTrainImages, augTrainLabels, validationSet

def trainNew(runName):

    # Get name for run
    runPath = './Runs/%s' % runName
    assert not os.path.exists(runPath), "Run name already exists, pick a new run name."
    os.makedirs(runPath)

    trainImages, trainLabels, validationSet = loadTrainingData()
    trainer.trainNew(runName, trainLabels, trainImages, validationSet)

def resume(runName, modelName):

    # Get model path
    curDir = os.getcwd()
    modelPath = "%s\\Runs\\%s\\Models\\%s.h5" % (curDir, runName, modelName)
    assert os.path.exists(modelPath), "Model does not exist."

    trainImages, trainLabels, validationSet = loadTrainingData()
    trainer.trainExisting(runName, modelPath, trainLabels, trainImages, validationSet)

def eval(runName, modelName):

    # Get model path
    curDir = os.getcwd()
    modelPath = "%s\\Runs\\%s\\Models\\%s.h5" % (curDir, runName, modelName)
    assert os.path.exists(modelPath), "Model does not exist."

    # Load Test Data
    print("Loading test data...", end="", flush=True)
    testIds, testImagesRaw = fileutils.read_test_data('./Data/test.csv')
    print("done.")

    # Normalize
    print("Normalizing data...", end="", flush=True)
    testImages = processing.normalizeImages(testImagesRaw)
    print("done.")

    # Load best
    print("Evaluating test set...")
    testLabels = trainer.evaluate(testImages, modelPath)

    print("Generating classification CSV...", end="", flush=True)
    fileutils.generate_classification(testIds, testLabels, runName)
    print("done.")

def checkPaths():
    fileutils.check_path("Data") # Folder for data
    fileutils.check_path("Runs") # Folder for training runs
    fileutils.check_path("Tensorboard") # Folder containing all tensorboard runs

if __name__ == "__main__":
    checkPaths()

    #resume('first', 'latest')
    #eval('ZeroValOldContrast64Batch', 'latest')
    trainNew(runName='ZeroValGood16Batch')