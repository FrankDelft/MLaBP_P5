# MLaBP_P5
# Image Compression with Autoencoders

This project explores the application of autoencoders for the purpose of image compression, using various neural network architectures. Autoencoders are a class of neural networks designed to compress the input data into a lower-dimensional representation and then reconstruct it back to its original form. This technique has significant applications in reducing image and video data size for efficient storage and transfer.




## Overview

We investigate multiple architectures, including the fully connected Autoencoder, Convolutional Autoencoders (CAE), Variational Autoencoders (VAE), and their traditional counterparts for compressing CIFAR-10 and Fashion-MNIST (FMNIST) datasets. The goal is to evaluate these models based on their compression efficiency and the quality of the reconstructed images.

## Getting Started

### Prerequisites

- Python 3.x
- TensorFlow/Keras
- PyTorch (for VAE models)

### Installation

1. Clone this repository to your local machine.
2. Install the required packages:
   ```bash
   pip install tensorflow keras torch matplotlib pandas
   ```
3. Create a Conda environment and activate it:
   ```bash
   conda create --name myenv python=3.x
   conda activate myenv
   ```

## Setup
In order to tun this code load the environment .yml file
To recreate the Conda environment, use the following command:

```bash
conda env create -f environment.yml
```

### Dataset

The project uses CIFAR-10 and Fashion-MNIST datasets, which can be automatically downloaded through the provided data loaders.

## Usage

The project is structured into several scripts:

- `data_loader.py`: Functions to load CIFAR-10 and FMNIST datasets.
- `ae_models.py`: Definitions of autoencoder models (CAE, VAE) for both datasets.
- `train_model.py`: Training procedures for Keras models.
- `train_torch_model.py`: Training procedures for PyTorch models.
- `visualizer.py`: Functions to visualize original and reconstructed images for comparison.
- `metrics.py`: Evaluation metrics for model performance.

To train and evaluate a model, run the corresponding script. For example, to train a Convolutional Autoencoder on FMNIST:

```bash
python train_model.py --model cae_fmnist
```

## Results

The project includes scripts for visualizing the performance of trained models on test data, comparing original and reconstructed images, and plotting loss curves.



