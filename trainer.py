import os
from subprocess import Popen

import keras
import psutil
from keras.layers import Conv2D, LeakyReLU, BatchNormalization, Dense, Flatten, SpatialDropout2D, MaxPooling2D
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import Adam, SGD

import fileutils


def build_network():
    # kernel_init = keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=None)
    # kernel_reg = keras.regularizers.l2(0.001)
    # br = keras.regularizers.l2(0.001)

    net = Sequential()
    net.add(Conv2D(64, kernel_size=5, input_shape=(28, 28, 1)))
    net.add(BatchNormalization(momentum=0.8))
    net.add(LeakyReLU(alpha=0.2))
    net.add(MaxPooling2D(pool_size=(2, 2)))
    net.add(SpatialDropout2D(0.25))

    net.add(Conv2D(32, kernel_size=3))
    net.add(BatchNormalization(momentum=0.8))
    net.add(LeakyReLU(alpha=0.2))
    net.add(MaxPooling2D(pool_size=(2, 2)))
    net.add(SpatialDropout2D(0.25))

    net.add(Flatten())
    net.add(Dense(128, activation=None))
    net.add(LeakyReLU(alpha=0.2))
    net.add(BatchNormalization(momentum=0.8))
    net.add(Dense(10, activation='softmax'))

    return net


def stop_tensorboard():
    # Check if tensorboard is already running and if it is, kill it
    for p in psutil.process_iter():
        if p.name() == "tensorboard.exe":
            p.kill()  # Kill tensorboard
            p.wait()  # Wait for it to terminate
            return


def start_tensorboard():
    # Check if tensorboard is already running
    for p in psutil.process_iter():
        if p.name() == "tensorboard.exe":
            return

    # Start new tensorboard instance
    current_path = os.getcwd()
    log_dir = '%s\\Tensorboard' % current_path
    Popen(['tensorboard', '--logdir=%s' % log_dir, '--host=localhost'], shell=True)


def delete_tensorboard_data(run_name):
    # Get tensorboard data path and delete all event files
    current_path = os.getcwd()
    base_dir = '%s\\Tensorboard\\%s\\' % (current_path, run_name)
    paths = os.listdir(base_dir)
    for path in paths:
        full_path = '%s\\%s' % (base_dir, path)
        if os.path.isfile(full_path):
            os.remove(full_path)


def train_new(run_name, train_labels, train_images, validation_set):
    # Create run in tensorboard directory
    tb_dir = "./Tensorboard/%s" % run_name
    os.makedirs(tb_dir)

    # Create Models directory
    models_dir = './Runs/%s/Models' % run_name
    os.makedirs(models_dir)

    # Create network and configure optimizer
    net = build_network()

    # Save network architecture
    yaml_str = net.to_yaml()
    fileutils.save_text('./Runs/%s/architecture.yaml' % run_name, yaml_str)

    # Compile new network with optimizer
    optimizer = Adam(lr=0.0001)
    net.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Train model
    train_model(run_name, net, train_labels, train_images, validation_set)


def train_existing(run_name, model_path, train_labels, train_images, validation_set):
    # Load network from file
    net = load_model(model_path)

    # Read last training epoch
    last_epoch = 0

    cur_dir = os.getcwd()
    epoch_data_file = "%s\\Runs\\%s\\training_epoch.dat" % (cur_dir, run_name)

    # Check if last epoch data exists
    if os.path.exists(epoch_data_file):

        # Try to parse data
        epoch_data = fileutils.read_text(epoch_data_file)
        if epoch_data is not None:
            last_epoch = int(epoch_data)

    # Test to see if we have the last epoch number
    if last_epoch == 0:
        # If we don't, lets delete the data so we can restart the tensorboard graph
        # First, make sure tensorboard is not running, so we can delete old data
        stop_tensorboard()

        # Delete old data because epochs are being reset to 0
        delete_tensorboard_data(run_name)

    # Train model
    train_model(run_name, net, train_labels, train_images, validation_set, last_epoch)


def train_model(run_name, net, train_labels, train_images, validation_set, epoch_start=1):
    # Get current working directory
    cur_dir = os.getcwd()

    # Get tensorboard directory
    tb_dir = "./Tensorboard/%s" % run_name

    # Get Models directory
    models_dir = './Runs/%s/Models' % run_name

    # Set batch size
    batch_size = 16

    # Create callback for saving latest epoch number
    save_epoch_callback = keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: fileutils.save_text("%s\\Runs\\%s\\training_epoch.dat" % (cur_dir, run_name),
                                                             str(epoch + 1)))

    # Create callback for automatically saving best model based on highest regular accuracy
    check_best_acc = keras.callbacks.ModelCheckpoint('%s/best_acc.h5' % models_dir, monitor='acc', verbose=0,
                                                     save_best_only=True, save_weights_only=False, mode='auto',
                                                     period=1)

    # Create callback for automatically saving best model based on highest validation accuracy
    check_best_val_acc = keras.callbacks.ModelCheckpoint('%s/best_val_acc.h5' % models_dir, monitor='val_acc',
                                                         verbose=0,
                                                         save_best_only=True, save_weights_only=False, mode='auto',
                                                         period=1)

    # Create callback for automatically saving lastest model so training can be resumed. Saves every epoch
    check_latest_callback = keras.callbacks.ModelCheckpoint('%s/latest.h5' % models_dir, verbose=0,
                                                            save_best_only=False,
                                                            save_weights_only=False, mode='auto', period=1)

    # Create callback for tensorboard
    tb_callback = keras.callbacks.TensorBoard(log_dir=tb_dir, batch_size=batch_size, write_graph=False,
                                              write_grads=True)

    # Create list of all callbacks, put least important callbacks (unstable ones that may fail) at end
    callback_list = [check_best_acc, check_latest_callback, tb_callback, save_epoch_callback, ]
    if validation_set is not None:
        callback_list = callback_list.insert(0, check_best_val_acc)  # Put at beginning of list

    # Start tensorboard
    start_tensorboard()

    # Train network and save best model along the way
    net.fit(train_images, train_labels, batch_size=batch_size, epochs=150, verbose=2, shuffle=True,
            validation_data=validation_set, callbacks=callback_list, initial_epoch=epoch_start)


def evaluate(test_images, model_path):
    # Load network from h5 format
    net = load_model(model_path)

    # Feed test images into network and get predictions
    onehot_predictions = net.predict(test_images)

    # Convert labels back from one-hot to single integer
    test_labels = [0] * onehot_predictions.shape[0]
    for i in range(onehot_predictions.shape[0]):
        highest_value = -1
        highest_index = -1
        for j in range(onehot_predictions.shape[1]):
            if onehot_predictions[i][j] > highest_value:
                highest_value = onehot_predictions[i][j]
                highest_index = j
        test_labels[i] = highest_index

    return test_labels
