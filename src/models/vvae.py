import numpy as np

import pytorch_lightning as pl
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F


class VanillaVAE(pl.LightningModule):

  def __init__(self, width, height, n_channels, n_hidden_units, n_latent):

    super().__init__()

    self.width = width
    self.height = height
    self.n_channels = n_channels

    self.activation = self.activations[activation]
    self.n_inputs_encoder = width * height * n_channels
    self.fc1 = nn.Linear(n_inputs_encoder, n_hidden_units)
    self.fc21 = nn.Linear(n_hidden_units, n_latent)
    self.fc22 = nn.Linear(n_hidden_units, n_latent)
    self.fc3 = nn.Linear(n_latent, n_hidden_units)
    self.fc4 = nn.Linear(n_hidden_units, n_hidden_units)

  def encode(self, x):
    h1 = F.relu(self.fc1(x))
    return self.fc21(h1), self.fc22(h1)

  def decode(self, x):
    h3 = F.relu(self.fc3(z))
    return torch.sigmoid(self.fc4(h3))

  def sample(self, mu, log_sigma2):
    eps = torch.randn(mu.shape[0], mu.shape[1])
    return mu + torch.exp(log_sigma2 / 2) * eps

  def forward(self, x):
    batch_size = x.shape[0]
    n_inputs_encoder = self.n_inputs_encoder

    # Want to flatten the input to
    # [batch_size, n_channels * width * height] for the encoder.
    x = x.view(batch_size, n_inputs_encoder)

    # Encode.
    mu, logstd = self.encode(x)
    # Sample.
    z = self.sample(mu, logstd)
    # Decode.
    reconstruction = self.decode(z)

    return reconstruction, mu, logstd

  def loss(self, x, reconstruction, mu, log_sigma2, sigma):
    recon = F.binary_cross_entropy(reconstruction, x,
                                   reduction='sum') / x.shape[0]
    kl = -0.5 * torch.mean(1 + log_sigma2 - mu.pow(2) - log_sigma2.exp())
    loss = recon + kl
    return loss
