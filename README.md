# FashionMnist
FashionMnist repository contains code to classify the fashion-mnist dataset through [kaggle](https://www.kaggle.com/) and 
their dataset. NOTE: This means we do not have the labels for the test set, which makes this project a little different than
most fashion mnist projects, since they can test their model much easier, we get 5 submissions per day and the accuracy result
is only from 30% of the actual test data. Right now the training is being done in python using keras and a tensorflow 
backend. Plans to experiment with Matlab's gpu computing toolkit are in the future. See below for more information on the project.

## Quick Links
- [Keras Documentation](https://keras.io/)
- [Competition Leaderboard](https://www.kaggle.com/c/uwb-css-485-fall-2018/leaderboard)

## Getting started
Here are some things you'll need to get started.

### Software and toolkits
1. [Cuda 9.0 Toolkit](https://developer.nvidia.com/cuda-90-download-archive?target_os=Windows&target_arch=x86_64)
2. [Python 3.6.2](https://www.python.org/downloads/release/python-362/)
3. [cuDNN v7.4.1 for CUDA 9.0](https://developer.nvidia.com/rdp/cudnn-download)
4. [Pycharm IDE](https://www.jetbrains.com/pycharm/download/)

### Python libraries
1. numpy: `pip3 install numpy`
2. tensorflow with gpu support: `pip3 install tensorflow-gpu`
3. keras: `pip3 install keras`
4. python imaging library: `pip3 install Pillow`
5. Process and system utilities library `pip3 install psutil`

### Downloading the dataset
The csv files are too large for us to put in the repository. 
1. Create a folder called **Data** 
2. Download the dataset [here](https://www.kaggle.com/c/10548/download-all)
3. Put train.csv and test.csv in the **Data** folder you created.