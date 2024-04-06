import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder_fmnist(nn.Module):
    def __init__(self, hidden_channels: int, latent_dim: int) -> None:
        super().__init__()
        # Input channels are 1 for grayscale images
        self.conv1 = nn.Conv2d(1, hidden_channels, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels*2, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        # After conv2, for a 28x28 image, the size will be 7x7
        self.conv3 = nn.Conv2d(hidden_channels*2, hidden_channels*4, 3, padding=1)
        # No need for a third pooling since the feature map would become too small
        # Flatten the output for the dense layers
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(hidden_channels*4*7*7, latent_dim)
        self.fc_logvar = nn.Linear(hidden_channels*4*7*7, latent_dim)
        
    def forward(self, x: torch.Tensor):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        return x_mu, x_logvar

class Decoder_fmnist(nn.Module):
    def __init__(self, hidden_channels: int, latent_dim: int) -> None:
        super().__init__()
        # The initial size will be 7x7 before unflattening
        self.fc = nn.Linear(latent_dim, hidden_channels*4*7*7)
        self.reshape = nn.Unflatten(1, (hidden_channels*4, 7, 7))
        self.conv3 = nn.ConvTranspose2d(hidden_channels*4, hidden_channels*2, 3, stride=1, padding=1)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')  # Upsample to 14x14
        self.conv2 = nn.ConvTranspose2d(hidden_channels*2, hidden_channels, 3, stride=1, padding=1)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')  # Upsample to 28x28
        # Output channels are 1 for grayscale images
        self.conv1 = nn.ConvTranspose2d(hidden_channels, 1, 3, stride=1, padding=1)
        
    def forward(self, z: torch.Tensor):
        z = F.relu(self.fc(z))
        z = self.reshape(z)
        z = F.relu(self.conv3(z))
        z = self.up1(z)
        z = F.relu(self.conv2(z))
        z = self.up2(z)
        z = torch.sigmoid(self.conv1(z))  # Use sigmoid for the last layer for grayscale values between 0 and 1
        return z

class VariationalAutoencoder_FMNIST(nn.Module):

    def __init__(self, hidden_channels: int, latent_dim: int):
        super().__init__()
        ###FILL IN: define an encoder by using the above Encoder class###
        self.encoder = Encoder_fmnist(hidden_channels=hidden_channels,
                                latent_dim=latent_dim)
        ###FILL IN:define a encoder by using the above Encoder class###
        self.decoder = Decoder_fmnist(hidden_channels=hidden_channels,
                                latent_dim=latent_dim)

    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar

    def latent_sample(self, mu, logvar):

        if self.training:
            # the reparameterization trick
            std = (logvar * 0.5).exp()
            return torch.distributions.Normal(loc=mu, scale=std).rsample()
            # std = logvar.mul(0.5).exp_()
            # eps = torch.empty_like(std).normal_()
            # return eps.mul(std).add_(mu)
        else:
            return mu
        
def vae_loss(recon_x, x, mu, logvar):

    recon_loss = F.mse_loss(recon_x.view(-1, 28*28*1), x.view(-1, 28*28*1), reduction='sum')
    ###FILL IN: KL Divergence Calculation###
    # You can look at the derivation of the KL term here https://arxiv.org/pdf/1907.08956.pdf
    kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss=recon_loss + kldivergence
    # return (total_loss,recon_loss)
    return total_loss, recon_loss

def refresh_bar(bar, desc):
    bar.set_description(desc)
    bar.refresh()


def make_averager() -> Callable[[Optional[float]], float]:
    """ Returns a function that maintains a running average

    :returns: running average function
    """
    count = 0
    total = 0

    def averager(new_value: Optional[float]) -> float:
        """ Running averager

        :param new_value: number to add to the running average,
                          if None returns the current average
        :returns: the current average
        """
        nonlocal count, total
        if new_value is None:
            return total / count if count else float("nan")
        count += 1
        total += new_value
        return total / count

    return averager