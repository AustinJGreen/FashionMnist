import os

from shutil import copyfile
import numpy as np
import tensorflow as tf

import fileutils
import processing
import trainer


# TODO: Auto submit csv file to kaggle with api
# TODO: Add config for all training which specifies all hyper-params
# TODO: Try transforms for vertical flips
# TODO: Try removing random box in each image
# TODO: Try increasing image size to 32x32
# TODO: Write function that averages weights between two models with same arch
# TODO: Try experimenting with normalizing weights between -1 and 1 or dividing by mean to create normal distribution


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


def submit_prediction(run_name, model_name):

    # Check for file
    cur_dir = os.getcwd()
    prediction_file = '%s\\Runs\\%s\\prediction.csv' % (cur_dir, run_name)
    assert os.path.exists(prediction_file), 'No submission file found.'

    kaggle_config = '%s\\kaggle.json' % cur_dir
    assert os.path.exists(kaggle_config), 'No kaggle API config found, create a API key in My Account.'

    # Check to make sure kaggle.json is in kaggle folder
    user = os.getlogin()
    kaggle_path = 'C:\\Users\\%s\\.kaggle\\' % user
    if not os.path.exists(kaggle_path):
        os.mkdir(kaggle_path)

    dst = '%s\\kaggle.json' % kaggle_path
    if not os.path.exists(dst):
        copyfile(kaggle_config, dst)

    print("Submitting prediction...")

    import kaggle

    client = kaggle.KaggleApi()
    client.authenticate()
    client.competition_submit(prediction_file, 'Run = %s, Model = %s' % (run_name, model_name), 'uwb-css-485-fall-2018')


def evaluate_and_submit(run_name, model_name):
    evaluate(run_name, model_name)
    submit_prediction(run_name, model_name)


def check_paths():
    fileutils.check_path("Data")  # Folder for data
    fileutils.check_path("Runs")  # Folder for training runs
    fileutils.check_path("Tensorboard")  # Folder containing all tensorboard runs


def set_seeds():
    np.random.seed(1337)
    tf.set_random_seed(1337)


if __name__ == "__main__":

    check_paths()
    set_seeds()

    # resume('DenseDropout_FT', 'latest')
    # train_new(run_name='DenseDropout_FT')
    # evaluate_and_submit('DenseDropout_FT', 'epoch050')
