"""
Module containing the encoders.
"""
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from disvae.utils.resblocks import ConvResBlock


# ALL encoders should be called Enccoder<Model>
def get_encoder(model_type):
    model_type = model_type.lower().capitalize()
    return eval("Encoder{}".format(model_type))


class EncoderBurgess(nn.Module):
    def __init__(self, img_size,
                 latent_dim=10):
        r"""Encoder of the model proposed in [1].

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).

        latent_dim : int
            Dimensionality of latent output.

        Model Architecture (transposed for decoder)
        ------------
        - 4 convolutional layers (each with 32 channels), (4 x 4 kernel), (stride of 2)
        - 2 fully connected layers (each of 256 units)
        - Latent distribution:
            - 1 fully connected layer of 20 units (log variance and mean for 10 Gaussians)

        References:
            [1] Burgess, Christopher P., et al. "Understanding disentangling in
            $\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
        """
        super(EncoderBurgess, self).__init__()

        # Layer parameters
        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256
        self.latent_dim = latent_dim
        self.img_size = img_size
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = self.img_size[0]

        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        self.conv1 = nn.Conv2d(n_chan, hid_channels, kernel_size, **cnn_kwargs)
        self.conv2 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.conv3 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)

        # If input image is 64x64 do fourth convolution
        if self.img_size[1] == self.img_size[2] == 64:
            self.conv_64 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)

        # Fully connected layers
        self.lin1 = nn.Linear(np.product(self.reshape), hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)

        # Fully connected layers for mean and variance
        self.mu_logvar_gen = nn.Linear(hidden_dim, self.latent_dim * 2)

    def forward(self, x):
        batch_size = x.size(0)

        # Convolutional layers with ReLu activations
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        if self.img_size[1] == self.img_size[2] == 64:
            x = torch.relu(self.conv_64(x))

        # Fully connected layers with ReLu activations
        x = x.view((batch_size, -1))
        x = torch.relu(self.lin1(x))
        x = torch.relu(self.lin2(x))

        # Fully connected layer for log variance and mean
        # Log std-dev in paper (bear in mind)
        mu_logvar = self.mu_logvar_gen(x)
        mu, logvar = mu_logvar.view(-1, self.latent_dim, 2).unbind(-1)

        return mu, logvar

class EncoderCustomres(nn.Module):
    def __init__(self, img_size, latent_dim=10):
        super(EncoderCustomres, self).__init__()

        self.input_dim = img_size
        self.latent_dim = latent_dim
        self.n_channels = 1

        self.conv_layers = nn.Sequential(
            ConvResBlock(1, 32),
            ConvResBlock(32, 64),
            ConvResBlock(64, 128),
            ConvResBlock(128, 256),
            ConvResBlock(256, 512),
            ConvResBlock(512, 512),
            ConvResBlock(512, 512),
            ConvResBlock(512, 1024)
        )
        self.lin1 = nn.Linear(1024, 512)
        self.lin2 = nn.Linear(512, 256)
        self.mu_log_var = nn.Linear(256, self.latent_dim*2)
        # self.log_var = nn.Linear(256, self.latent_dim)

    def forward(self, x: torch.Tensor):
        h1 = self.conv_layers(x).reshape(x.shape[0], -1)
        x = F.leaky_relu(self.lin1(h1))
        x = F.leaky_relu(self.lin2(x))
        mu_logvar = self.mu_log_var(x)
        mu, logvar = mu_logvar.view(-1, self.latent_dim, 2).unbind(-1)
        # logvar = self.log_var(self.lin2(self.lin1(h1)))
        return mu, logvar
