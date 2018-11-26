import os

import numpy as np

import fileutils
import processing
import trainer

# TODO: Auto submit csv file to kaggle with api
# TODO: Add config for all training which specifies all hyper-params
# TODO: Save more models (e.g. for loss)


def load_training_data():
    # Load Training Data
    print("Reading training data...", end="", flush=True)
    _, y_train, x_train = fileutils.read_train_data_raw('./Data/train.csv')
    print("done.")

    # Normalize
    print("Normalizing data...", end="", flush=True)
    train_images = processing.normalize_images(x_train)
    train_labels = processing.convert_labels(y_train)
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
        train_images = train_images[:-validation_set_size]
        train_labels = train_labels[:-validation_set_size]

    print("done.")

    # Run Test
    print("Generating augmented training set...", end="", flush=True)
    aug_train_images, aug_train_labels = processing.augment_images(train_images, train_labels)
    print("done.")

    # Shuffle data very well
    print("Shuffling augmented data...", end="", flush=True)
    aug_train_images, aug_train_labels = processing.shuffle(aug_train_images, aug_train_labels)
    print("done.")

    return aug_train_images, aug_train_labels, validation_set


def train_new(run_name):

    # Get name for run
    run_path = './Runs/%s' % run_name
    assert not os.path.exists(run_path), "Run name already exists, pick a new run name."
    os.makedirs(run_path)

    train_images, train_labels, validation_set = load_training_data()
    trainer.train_new(run_name, train_labels, train_images, validation_set)


def resume(run_name, model_name):

    # Get model path
    cur_dir = os.getcwd()
    model_path = "%s\\Runs\\%s\\Models\\%s.h5" % (cur_dir, run_name, model_name)
    assert os.path.exists(model_path), "Model does not exist."

    train_images, train_labels, validation_set = load_training_data()
    trainer.train_existing(run_name, model_path, train_labels, train_images, validation_set)


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
    test_images = processing.normalize_images(test_images_raw)
    print("done.")

    # Load best
    print("Evaluating test set...")
    test_labels = trainer.evaluate(test_images, model_path)

    print("Generating classification CSV...", end="", flush=True)
    fileutils.generate_classification(test_ids, test_labels, run_name)
    print("done.")


def check_paths():
    fileutils.check_path("Data")  # Folder for data
    fileutils.check_path("Runs")  # Folder for training runs
    fileutils.check_path("Tensorboard")  # Folder containing all tensorboard runs


if __name__ == "__main__":
    check_paths()

    # resume('94RunMoreDropout', 'latest')
    # evaluate('94RunMoreDropout', 'latest')
    train_new(run_name='Resnet1')
