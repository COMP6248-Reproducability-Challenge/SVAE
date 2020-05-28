import numpy as np

import pytorch_lightning as pl
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F


class VanillaVAE(pl.LightningModule):

  def __init__(self,
               width,
               height,
               n_channels,
               n_hidden_units,
               n_latent,
               activation=nn.Tanh()):

    super().__init__()

    self.width = width
    self.height = height
    self.n_channels = n_channels

    self.activation = activation
    self.n_inputs_encoder = width * height * n_channels
    self.f_enc_input = nn.Linear(self.n_inputs_encoder, n_hidden_units)
    self.f_enc_mu = nn.Linear(n_hidden_units, n_latent)
    self.f_enc_logvar = nn.Linear(n_hidden_units, n_latent)
    self.f_dec_input = nn.Linear(n_latent, n_hidden_units)
    self.f_dec_output = nn.Linear(n_hidden_units, self.n_inputs_encoder)

  def encode(self, x):
    h1 = self.activation(self.f_enc_input(x))
    return self.f_enc_mu(h1), self.f_enc_logvar(h1)

  def decode(self, z):
    batch_size = z.shape[0]
    n_channels = self.n_channels
    width = self.width
    height = self.height
    h3 = self.activation(self.f_dec_input(z))
    return torch.sigmoid(self.f_dec_output(h3)).view(batch_size, n_channels,
                                                     width, height)

  def sample(self, mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

  def forward(self, x):
    batch_size = x.shape[0]
    n_inputs_encoder = self.n_inputs_encoder

    # Want to flatten the input to
    # [batch_size, n_channels * width * height] for the encoder.
    x = x.view(batch_size, n_inputs_encoder)

    # Encode.
    mu, logvar = self.encode(x)
    # Sample.
    z = self.sample(mu, logvar)
    # Decode.
    reconstruction = self.decode(z)

    return reconstruction, mu, logvar

  def loss(self, x, reconstruction, mu, logvar):
    recon = F.binary_cross_entropy(reconstruction, x,
                                   reduction='sum') / x.shape[0]
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)
    loss = recon + kl.mean()
    return loss
