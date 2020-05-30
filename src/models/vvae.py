import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F


class VanillaVAE(pl.LightningModule):

  def __init__(self,
               width,
               height,
               n_channels,
               n_hidden_units,
               n_hidden,
               n_unconstrained,
               activation=nn.Tanh()):

    super().__init__()

    self.width = width
    self.height = height
    self.n_channels = n_channels
    self.n_hidden_units = n_hidden_units
    self.n_hidden = n_hidden
    self.n_unconstrained = n_unconstrained
    self.activation = activation

    self.n_pixels = width * height * n_channels

    encoder_layers = []
    # Input layer.
    encoder_layers.append(nn.Linear(self.n_pixels, n_hidden_units))
    encoder_layers.append(self.activation)
    # Hidden layers.
    for _ in range(n_hidden):
      encoder_layers.append(nn.Linear(n_hidden_units, n_hidden_units))
      encoder_layers.append(self.activation)
    # Output layer.
    encoder_layers.append(nn.Linear(n_hidden_units, 2 * self.n_unconstrained))
    self.encoder = nn.Sequential(*encoder_layers)

    decoder_layers = []
    # Input layer.
    decoder_layers.append(nn.Linear(self.n_unconstrained, n_hidden_units))
    decoder_layers.append(self.activation)
    # Hidden layers.
    for _ in range(n_hidden):
      decoder_layers.append(nn.Linear(n_hidden_units, n_hidden_units))
      decoder_layers.append(self.activation)
    # Output layer.
    decoder_layers.append(nn.Linear(n_hidden_units, self.n_pixels))
    decoder_layers.append(nn.Sigmoid())
    self.decoder = nn.Sequential(*decoder_layers)

  def encode(self, x):
    encoded = self.encoder(x)
    mu, logvar = encoded.split(self.n_unconstrained, 1)
    return mu, logvar

  def sample(self, mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

  def decode(self, z):
    batch_size = z.shape[0]
    n_channels = self.n_channels
    width = self.width
    height = self.height
    decoded = self.decoder(z).view(batch_size, n_channels, width, height)
    return decoded

  def forward(self, x):
    batch_size = x.shape[0]
    n_pixels = self.n_pixels

    x = x.view(batch_size, n_pixels)

    # Encode.
    mu, logvar = self.encode(x)
    # Sample.
    z = self.sample(mu, logvar)
    # Decode.
    reconstruction = self.decode(z)

    return reconstruction, mu, logvar

  def loss(self, x, reconstruction, mu, logvar):
    batch_size = x.shape[0]
    reconstruction_loss = F.binary_cross_entropy(
        reconstruction, x, reduction='sum') / batch_size

    kl_div = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)
    loss = reconstruction_loss + kl_div.mean()
    return loss
