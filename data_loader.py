from keras.datasets import cifar10, fashion_mnist
import numpy as np
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
from torchvision.datasets import CIFAR10
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

CIFAR10_label_txts = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
MNIST_label_txts = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


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


img_transform = transforms.Compose([
    transforms.ToTensor()
])

def get_CIFAR_torch():
    batch_size = 128
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=img_transform)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=img_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return(train_dataloader,test_dataloader)


def get_FMNST_torch():
    batch_size = 128
    train_dataset = FashionMNIST(root='./data', train=True, download=True, transform=img_transform)
    test_dataset = FashionMNIST(root='./data', train=False, download=True, transform=img_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return(train_dataloader,test_dataloader)
