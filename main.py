import os

from shutil import copyfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import fileutils
import processing
import trainer

# TODO: Write function that averages weights between two models with same arch
# TODO: Try experimenting with normalizing weights between -1 and 1 or dividing by mean to create normal distribution

def matrix_compare(a, b):
    w = a.shape[0]
    h = a.shape[1]

    for x in range(w):
        for y in range(h):
            if a[x, y] != b[x, y]:
                return False
    return True


def load_training_data():
    # Load Training Data
    print("Reading training data...", end="", flush=True)
    _, y_train, x_train = fileutils.read_train_data_raw('./Data/train.csv')
    print("done.")

    # Normalize
    # 0 T-shirt/top
    # 1 Trouser
    # 2 Pullover
    # 3 Dress
    # 4 Coat
    # 5 Sandal
    # 6 Shirt
    # 7 Sneaker
    # 8 Bag
    # 9 Ankle boot

    print("Normalizing data...", end="", flush=True)
    reshaped_images = processing.reshape_images(x_train)
    # resized_images = processing.resize_images(reshaped_images, 64)
    train_images = processing.normalize_images(reshaped_images)
    train_labels = processing.convert_labels(y_train, categories=10, single=None)
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


def evaluate_single(run_name, model_name):
    # Get model path
    cur_dir = os.getcwd()
    model_path = "%s\\Runs\\%s\\Models\\%s.h5" % (cur_dir, run_name, model_name)
    assert os.path.exists(model_path), "Model does not exist."

    # Load Test Data
    print("Loading test data...", end="", flush=True)
    test_ids, test_images_raw = fileutils.read_test_data_raw('./Data/test.csv')
    print("done.")

    # Normalize
    print("Normalizing data...", end="", flush=True)
    reshaped_images = processing.reshape_images(test_images_raw)
    test_images = processing.normalize_images(reshaped_images)
    print("done.")

    # Load best
    print("Evaluating test set...")
    _, test_labels = trainer.evaluate(test_images, model_path)

    print("Generating classification CSV...", end="", flush=True)
    fileutils.generate_classification(test_ids, test_labels, run_name)
    print("done.")


def evaluate_multiple(run_names, model_names, categories, backup_model):
    runs_len = len(run_names)
    models_len = len(model_names)
    categories_len = len(categories)

    assert runs_len == models_len and models_len == categories_len, "Runs, models, and categories must be the same length."

    # Get model paths
    cur_dir = os.getcwd()
    model_paths = list()
    for i in range(runs_len):
        run_name = run_names[i]
        model_name = model_names[i]
        model_path = "%s\\Runs\\%s\\Models\\%s.h5" % (cur_dir, run_name, model_name)
        assert os.path.exists(model_path), "Model for %s does not exist." % run_name
        model_paths.append(model_path)

    # Load Test Data
    print("Loading test data...", end="", flush=True)
    test_ids, test_images_raw = fileutils.read_test_data_raw('./Data/test.csv')
    print("done.")

    # Normalize
    print("Normalizing data...", end="", flush=True)
    test_images = processing.normalize_images(test_images_raw)
    print("done.")

    # Load best
    print("Creating evaluations...")
    eval_labels = list()
    for i in range(runs_len):
        predictions, _ = trainer.evaluate(test_images, model_paths[i])
        print("Creating evaluations... (%s/%s)" % (i + 1, runs_len))
        eval_labels.append(predictions)

    print("Creating backup evaluations...")
    backup_confidence, backup_predictions = trainer.evaluate(test_images, backup_model)
    print("Generating labels...")
    test_predictions = [0] * test_ids.shape[0]
    for j in range(len(test_predictions)):

        # Check each prediction and get highest confidence, also count conflicts
        conflicts = 0
        highest_confidence = -1
        highest_confidence_label = -1
        for k in range(runs_len):
            cur_conf = eval_labels[k][j][0]
            cur_not_conf = eval_labels[k][j][1]
            if cur_conf > cur_not_conf and cur_conf > highest_confidence:
                highest_confidence = cur_conf
                highest_confidence_label = categories[k]
                if highest_confidence_label != -1:
                    conflicts = conflicts + 1

        # Maybe do something with predictions based off of conflicts
        if highest_confidence_label == -1:
            # None of the invidual models think its that classification, refer to backup model
            highest_confidence_label = backup_predictions[j]
            highest_confidence = backup_confidence[k][highest_confidence_label]

        test_predictions[j] = highest_confidence_label

    print("Generating classification CSV...", end="", flush=True)
    fileutils.generate_classification(test_ids, test_predictions, run_names[0])
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
    evaluate_single(run_name, model_name)
    submit_prediction(run_name, model_name)


def check_paths():
    fileutils.check_path("Data")  # Folder for data
    fileutils.check_path("Runs")  # Folder for training runs
    fileutils.check_path("Tensorboard")  # Folder containing all tensorboard runs


def set_seeds():
    np.random.seed(1337)
    tf.set_random_seed(1337)


def generate_matlab_plot(csv_file, title):
    data0 = fileutils.read_csv_data(csv_file, float)
    x_data0 = data0[:, 1]
    y_data0 = data0[:, 2]

    plt.plot(x_data0, y_data0)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.title(title)
    plt.savefig('C:\\Users\\austi\\Desktop\\plot.png')


if __name__ == "__main__":

    check_paths()

