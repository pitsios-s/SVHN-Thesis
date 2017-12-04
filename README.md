# SVHN-Thesis
This repository extends [https://github.com/pitsios-s/SVHN](https://github.com/pitsios-s/SVHN), so that it can be
used for the M.Sc. thesis of [Business Analytics Program](http://analytics.aueb.gr/).

Street View House Numbers (SVHN) is a real-world dataset containing images of house numbers taken from Google's street view. This repository contains the source code needed to built machine learning algorithms that can recognize and predict house numbers on scenery images.

The dataset comes in 2 different formats, full numbers which includes numbers with up to 6 digits in length and cropped images, which is used for single digit prediction, similar to [MNIST](http://yann.lecun.com/exdb/mnist/). In this repository we provide all the necessary code to address both problems.

To achieve our goal, a deep neural network was developed, and especially a Convolutional Neural Network (CNN). The implementation is based on TensorFlow.

# Running the Code

## Necessary Libraries
For the code to run, the following libraries are needed which can be installed via pip:
* Numpy: `pip install numpy`
* Scipy: `pip install scipy`
* Matplotlib: `pip install matplotlib`
* H5py: `pip install h5py`
* Pillow: `pip install Pillow`
* Tensorflow: `pip install tensorflow` For CPU only, or `pip install tensorflow-gpu` for TensorFlow with GPU support (also needs CUDA and cuDNN installation)
