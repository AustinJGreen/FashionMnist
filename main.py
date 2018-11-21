import trainer
import fileutils
import processing
import numpy as np
import os


# TODO: Auto submit csv file to kaggle with api
# TODO: Separate runs in folder directories and show all on tensorboard


def load_training_data():
    # Load Training Data
    print("Reading training data...", end="", flush=True)
    _, y_train, x_train = fileutils.read_train_data_raw('./Data/train.csv')
    print("done.")

    # Normalize
    print("Normalizing data...", end="", flush=True)
    train_images = processing.normalizeImages(x_train)
    train_labels = processing.convertLabels(y_train)
    print("done.")

    # Shuffle data very well
    print("Shuffling training data...", end="", flush=True)
    train_images, train_labels = processing.shuffle(train_images, train_labels)
    print("done.")

    print("Generating validation set...", end="", flush=True)
    validation_set_size = int(0 * train_images.shape[0])
    validation_set = None
    if validation_set_size > 0:
        validation_set = (train_images[-validation_set_size:], train_labels[-validation_set_size:])

        # TODO: Test distribution
        max_sums = np.sum(train_labels, axis=0)

        train_images = train_images[:-validation_set_size]
        train_labels = train_labels[:-validation_set_size]

        label_sums = np.sum(train_labels, axis=0)
    print("done.")

    # Run Test
    print("Generating augmented training set...", end="", flush=True)
    aug_train_images, aug_train_labels = processing.augmentImages(train_images, train_labels)
    print("done.")

    # Shuffle data very well
    print("Shuffling augmented data...", end="", flush=True)
    aug_train_images, aug_train_labels = processing.shuffle(aug_train_images, aug_train_labels)
    print("done.")

    return aug_train_images, aug_train_labels, validation_set


def train_new(runName):

    # Get name for run
    run_path = './Runs/%s' % runName
    assert not os.path.exists(run_path), "Run name already exists, pick a new run name."
    os.makedirs(run_path)

    train_images, train_labels, validation_set = load_training_data()
    trainer.trainNew(runName, train_labels, train_images, validation_set)


def resume(runName, modelName):

    # Get model path
    cur_dir = os.getcwd()
    model_path = "%s\\Runs\\%s\\Models\\%s.h5" % (cur_dir, runName, modelName)
    assert os.path.exists(model_path), "Model does not exist."

    train_images, train_labels, validation_set = load_training_data()
    trainer.trainExisting(runName, model_path, train_labels, train_images, validation_set)


def evaluate(run_name, model_name):

    # Get model path
    cur_dir = os.getcwd()
    model_path = "%s\\Runs\\%s\\Models\\%s.h5" % (cur_dir, run_name, model_name)
    assert os.path.exists(model_path), "Model does not exist."

    # Load Test Data
    print("Loading test data...", end="", flush=True)
    test_ids, test_images_raw = fileutils.read_test_data('./Data/test.csv')
    print("done.")

    # Normalize
    print("Normalizing data...", end="", flush=True)
    test_images = processing.normalizeImages(test_images_raw)
    print("done.")

    # Load best
    print("Evaluating test set...")
    test_labels = trainer.evaluate(test_images, model_path)

    print("Generating classification CSV...", end="", flush=True)
    fileutils.generate_classification(test_ids, test_labels, run_name)
    print("done.")


def check_paths():
    fileutils.check_path("Data") # Folder for data
    fileutils.check_path("Runs") # Folder for training runs
    fileutils.check_path("Tensorboard") # Folder containing all tensorboard runs


if __name__ == "__main__":
    check_paths()

    #resume('first', 'latest')
    #eval('ZeroValOldContrast64Batch', 'latest')
    train_new(runName='ZeroValGood16Batch')