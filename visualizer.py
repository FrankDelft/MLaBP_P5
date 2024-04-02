# Embeddings/Visualization of latant variables
from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from ae_models import torch_predict
from tqdm import tqdm
import torch

def plot_all(data, components:int):

    umap_reducer = UMAP(n_components=components, n_neighbors=15, min_dist=0.1)
    tsne_reducer = TSNE(n_components=components, perplexity=3)
    pca_reducer  = PCA(n_components=components)

    UMAP_embedding = umap_reducer.fit_transform(data)
    print(UMAP_embedding.shape)

    TSNE_embedding = tsne_reducer.fit_transform(data)
    print(TSNE_embedding.shape)

    PCA_embedding = pca_reducer.fit_transform(data)
    print(PCA_embedding.shape)

    plt.plot(UMAP_embedding)
    plt.plot(TSNE_embedding)
    plt.plot(PCA_embedding)

def compare_CIFAR(x_test, autoencoder,*args):
    if len(args) > 0:
        images = torch_predict(autoencoder, args[0])
        
    else:
        images = autoencoder.predict(x_test)
    # Select 10 indices
    indices = np.arange(10, 20)

    # Plot the input vs output images
    fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(20, 4))

    for i, idx in enumerate(indices):
        # Plot input image
        axes[0, i].imshow(x_test[idx].reshape(32, 32, 3))
        axes[0, i].axis('off')

        # Plot output image
        axes[1, i].imshow(images[idx].reshape(32, 32, 3))
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()


def compare_FMNIST(x_test, autoencoder,*args):
    if len(args) > 0:
        decoded_imgs = torch_predict(autoencoder, args[0])

    else:
        decoded_imgs = autoencoder.predict(x_test)

    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(0, n):
        # Display original
        ax = plt.subplot(2, n, i+1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.axis('off')

        # Display reconstruction
        ax = plt.subplot(2, n, i+1+n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def compare_fmnist_models(models, image,*args):
    n = len(models)+1
    image = image.reshape(1,28,28,1)
    plt.figure(figsize=(2*n, 2))

    ax = plt.subplot(1, n, 1)
    plt.imshow(image.reshape(28, 28))
    plt.gray()
    plt.title('Original')
    ax.axis('off')

    for i, m in enumerate(models):
        # Display original
        if(i<len(models)-1):
            decoded_img =  m[0].predict(image.reshape(-1,3072) if m[2] else image)
        else:
            decoded_img=torch_predict(m[0],args[0])
            decoded_img=decoded_img[args[1]+3]
        decoded_img =  m[0].predict(image.reshape(-1,784) if m[2] else image)

        # Display reconstruction
        ax = plt.subplot(1, n, i+2)
        plt.imshow(decoded_img.reshape(28, 28))
        plt.gray()
        plt.title(m[1])
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def compare_cifar_models(models, image,*args):
    n = len(models)+1
    image = image.reshape(1,32,32,3)
    plt.figure(figsize=(2*n, 2))

    ax = plt.subplot(1, n, 1)
    plt.imshow(image.reshape(32, 32, 3))
    plt.gray()
    plt.title('Original')
    ax.axis('off')

    for i, m in enumerate(models):
        # Display original
        if(i<len(models)-1):
            decoded_img =  m[0].predict(image.reshape(-1,3072) if m[2] else image)
        else:
            decoded_img=torch_predict(m[0],args[0])
            decoded_img=decoded_img[args[1]+3]
        
        
        # Display reconstruction
        ax = plt.subplot(1, n, i+2)
        plt.imshow(decoded_img.reshape(32, 32, 3))
        plt.gray()
        plt.title(m[1])
        ax.axis('off')
    plt.tight_layout()
    plt.show()