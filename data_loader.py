from keras.datasets import cifar10, fashion_mnist
import numpy as np
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

def get_cifar10():
    (x_train, _), (x_test, labels_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 32, 32, 3))
    x_test = np.reshape(x_test, (len(x_test), 32, 32, 3))
    return (x_train, x_test, labels_test)

def get_FMNIST():
    (x_train, _), (x_test, labels_test) = fashion_mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
    return (x_train, x_test, labels_test)