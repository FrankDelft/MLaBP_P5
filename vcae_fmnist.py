import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Callable, Optional

class Encoder_FMNIST(nn.Module):
    def __init__(self, hidden_channels: int, latent_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=hidden_channels,
                               kernel_size=4,
                               stride=2,
                               padding=1) # out: hidden_channels x 14 x 14

        self.conv2 = nn.Conv2d(in_channels=hidden_channels,
                               out_channels=hidden_channels*2,
                               kernel_size=4,
                               stride=2,
                               padding=1) # out: (hidden_channels x 2) x 7 x 7

        self.conv3 = nn.Conv2d(in_channels=hidden_channels*2,
                               out_channels=hidden_channels*4,
                               kernel_size=3,
                               padding=1) # out: (hidden_channels x 4) x 7 x 7

        self.fc_mu = nn.Linear(in_features=hidden_channels*4*7*7,
                               out_features=latent_dim)
        self.fc_logvar = nn.Linear(in_features=hidden_channels*4*7*7,
                                   out_features=latent_dim)

        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))

        x = x.view(x.shape[0], -1)

        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)

        return x_mu, x_logvar

class Decoder_FMNIST(nn.Module):
    def __init__(self, hidden_channels: int, latent_dim: int) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels

        self.fc = nn.Linear(in_features=latent_dim,
                            out_features=hidden_channels*4*7*7)

        self.conv3 = nn.ConvTranspose2d(in_channels=hidden_channels*4,
                                        out_channels=hidden_channels*2,
                                        kernel_size=3,
                                        padding=1)

        self.conv2 = nn.ConvTranspose2d(in_channels=hidden_channels*2,
                                        out_channels=hidden_channels,
                                        kernel_size=4,
                                        stride=2,
                                        padding=1)

        self.conv1 = nn.ConvTranspose2d(in_channels=hidden_channels,
                                        out_channels=1,
                                        kernel_size=4,
                                        stride=2,
                                        padding=1)

        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = x.view(x.size(0), self.hidden_channels*4, 7, 7)
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv2(x))
        x = torch.sigmoid(self.conv1(x)) # last layer before output is sigmoid
        return x
    

class VariationalAutoencoder_FMNIST(nn.Module):

    def __init__(self, hidden_channels: int, latent_dim: int):
        super().__init__()
        ###FILL IN: define an encoder by using the above Encoder class###
        self.encoder = Encoder_FMNIST(hidden_channels=hidden_channels,
                                latent_dim=latent_dim)
        ###FILL IN:define a encoder by using the above Encoder class###
        self.decoder = Decoder_FMNIST(hidden_channels=hidden_channels,
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

    recon_loss = F.mse_loss(recon_x.view(-1, 32*32*3), x.view(-1, 32*32*3), reduction='sum')
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