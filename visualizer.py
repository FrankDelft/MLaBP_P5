# Embeddings/Visualization of latant variables
from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from ae_models import torch_predict
from tqdm import tqdm
import torch

def plot_all(data, components:int, neighbors=15, min_distance=0.1, perplex=3):
    """
    Plots the embeddings of the input data using UMAP, t-SNE, and PCA dimensionality reduction techniques.

    Parameters:
    - data: The input data to be visualized.
    - components: The number of components in the reduced embedding space.
    - neighbors: The number of nearest neighbors to consider for UMAP.
    - min_distance: The minimum distance between points in the UMAP embedding.
    - perplex: The perplexity parameter for t-SNE.

    Returns:
    None
    """

    umap_reducer = UMAP(n_components=components, n_neighbors=neighbors, min_dist=min_distance)
    tsne_reducer = TSNE(n_components=components, perplexity=perplex)
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

def plot_pca(latent, labels, labels_text=None, name="PCA"):
    """
    Plot the Principal Component Analysis (PCA) visualization of the given latent space.

    Parameters:
    - latent (numpy.ndarray): The latent space array.
    - labels (numpy.ndarray): The labels corresponding to each data point in the latent space.
    - labels_text (list, optional): The text labels for each unique label. If not provided, unique labels will be used.

    Returns:
    None
    """
    pca_reducer = PCA(n_components=2)
    PCA_embedding = pca_reducer.fit_transform(latent)
    plt.scatter(PCA_embedding[:, 0], PCA_embedding[:, 1], c=labels, cmap='viridis', marker='.', s=2)
    plt.title("Latent PCA")

    # Get unique labels and their respective colors
    unique_labels = np.unique(labels)
    if labels_text is None:
        labels_text = unique_labels.tolist()
    colors = [plt.cm.viridis(i / float(len(unique_labels) - 1)) for i in range(len(unique_labels))]

    # Create legend outside of the subplot grid
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=label, markersize=10, markerfacecolor=color) for
               label, color in zip(unique_labels, colors)]
    plt.legend(handles=handles, labels=labels_text, title='Labels', loc='center right')
    plt.rcParams.update({'font.size': 11})
    plt.savefig(f"./figures/{name}.pdf")

def plot_single_umap(latent, labels, n_neighbors, min_dist, name="UMAP", labels_text=None):
    """
    Plot a UMAP visualization of the given latent space.

    Parameters:
    - latent (numpy.ndarray): The latent space array.
    - labels (numpy.ndarray): The labels for each data point in the latent space.
    - n_neighbors (int): The number of neighbors to consider for each point.
    - min_dist (float): The minimum distance between points in the UMAP embedding.
    - name (str, optional): The name of the plot. Defaults to "UMAP".
    - labels_text (list, optional): The text labels for each unique label. Defaults to None.

    Returns:
    None
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    umap_reducer = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist)
    umap_embedding = umap_reducer.fit_transform(latent)
    ax.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=labels, cmap='viridis', marker='.', s=2)
    ax.set_title(f'UMAP:num neigh={n_neighbors}, min dist={min_dist}')

    # Get unique labels and their respective colors
    unique_labels = np.unique(labels)
    if labels_text == None:
        labels_text = unique_labels.tolist()
    colors = [plt.cm.viridis(i / float(len(unique_labels) - 1)) for i in range(len(unique_labels))]

    # Create legend outside of the subplot grid
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=label, markersize=10, markerfacecolor=color) for
               label, color in zip(unique_labels, colors)]
    fig.legend(handles=handles, labels=labels_text, title='Labels', loc='lower right')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95, right=0.85)  # Adjust top and right margin to make space for the legend and title
    plt.rcParams.update({'font.size': 18})
    plt.savefig(f"./figures/{name}.pdf")

def plot_umap_grid(latent, labels, neighbors_list=[5, 20, 50, 100], min_dist_list=[0.1, 0.5, 1], name="UMAP_grid", labels_text=None):
    """
    Plots a grid of UMAP embeddings with different values of neighbors and min_dist.

    Parameters:
    - latent (numpy.ndarray): The latent space representation of the data.
    - labels (numpy.ndarray): The labels for each data point.
    - neighbors_list (list): A list of integers representing the number of neighbors for UMAP. Default is [5, 20, 50, 100].
    - min_dist_list (list): A list of floats representing the minimum distance for UMAP. Default is [0.1, 0.5, 1].
    - name (str): The name of the output file. Default is "UMAP_grid".
    - labels_text (list): A list of strings representing the labels for the legend. Default is None.

    Returns:
    - None
    """

    fig, axs = plt.subplots(len(neighbors_list), len(min_dist_list), figsize=(30, 30))

    # Set titles for each column based on min_dist values
    for ax, col in zip(axs[0], min_dist_list):
        ax.set_title(f'Min Dist: {col}')

    # Set labels for each row based on neighbors values
    for ax, row in zip(axs[:, 0], neighbors_list):
        ax.set_ylabel(f'Neighbors: {row}', rotation=90, size='large')

    # Generate UMAP embeddings for each combination of neighbors and min_dist
    for indn, n in enumerate(neighbors_list):
        for indd, d in enumerate(min_dist_list):
            umap_reducer = UMAP(n_components=2, n_neighbors=n, min_dist=d)
            UMAP_embedding = umap_reducer.fit_transform(latent)
            axs[indn, indd].scatter(UMAP_embedding[:, 0], UMAP_embedding[:, 1], c=labels, cmap='viridis', marker='.', s=1)

    # Get unique labels and their respective colors
    unique_labels = np.unique(labels)
    if labels_text is None:
        labels_text = unique_labels.tolist()
    colors = [plt.cm.viridis(i / float(len(unique_labels) - 1)) for i in range(len(unique_labels))]

    # Create legend outside of the subplot grid
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=label, markersize=10, markerfacecolor=color) for
               label, color in zip(unique_labels, colors)]
    fig.legend(handles=handles, labels=labels_text, title='Labels', loc='center right')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95, right=0.95)  # Adjust top and right margin to make space for the legend and title

    plt.savefig(f"./figures/{name}.pdf")

def plot_TSNE_grid(latent, labels, p_start = 0.1, p_end = 100, p_num = 15, name = "TSNE_grid", labels_text = None):
    """
    Plots a grid of t-SNE visualizations with varying perplexity values.

    Parameters:
    - latent (numpy.ndarray): The latent space representation of the data.
    - labels (numpy.ndarray): The labels for each data point.
    - p_start (float): The starting perplexity value for the grid. Default is 0.1.
    - p_end (float): The ending perplexity value for the grid. Default is 100.
    - p_num (int): The number of perplexity values to include in the grid. Default is 15.
    - name (str): The name of the output file. Default is "TSNE_grid".
    - labels_text (list): The text labels for each unique label. If None, the unique labels will be used. Default is None.

    Returns:
    None
    """
    perp_list = np.logspace(np.log10(p_start), np.log10(p_end), num=p_num)
    fig, axs = plt.subplots(int(np.ceil(len(perp_list)/3)), 3, figsize=(30, 30))
    fig.suptitle('TSNE Grid')

    for indp, p in enumerate(perp_list):
        tsne_reducer = TSNE(n_components=2, perplexity=p)
        tsne_embedding = tsne_reducer.fit_transform(latent)
        ax = axs[int(indp/3), int(indp%3)]
        ax.scatter(tsne_embedding[:,0], tsne_embedding[:,1], c=labels, cmap='viridis', marker='.', s=1)
        ax.set_title(f'Perplexity: {p:.2f}')

    # Get unique labels and their respective colors
    unique_labels = np.unique(labels)
    if labels_text == None: labels_text = unique_labels.tolist()
    colors = [plt.cm.viridis(i/float(len(unique_labels)-1)) for i in range(len(unique_labels))]

    # Create legend outside of the subplot grid
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=label, markersize=10, markerfacecolor=color) for label, color in zip(unique_labels, colors)]
    fig.legend(handles=handles, labels=labels_text, title='Labels', loc='center right')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95, right=0.95)  # Adjust top and right margin to make space for the legend and title

    plt.savefig(f"./figures/{name}.pdf")


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
    plt.tight_layout()

def compare_FMNIST(x_test, autoencoder,*args):
    if len(args) > 0:
        decoded_imgs = torch_predict(autoencoder, args[0],fmnist=True)

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
            decoded_img =  m[0].predict(image.reshape(-1,784) if m[2] else image)
        else:
            decoded_img=torch_predict(m[0],args[0],fmnist=True)
            decoded_img=decoded_img[args[1]+3]


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