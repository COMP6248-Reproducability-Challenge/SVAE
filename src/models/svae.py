import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F


def make_normalized_grid(width, height):
  """Generate a normalized grid.

  Args:
  width: int
  height:int

  Returns:
  grid: torch.tensor
    Shape [width, height, 3].
  """
  x = np.linspace(-1, 1, width)
  y = np.linspace(1, -1, height)
  xv, yv = np.meshgrid(x, y)
  ones = np.ones((width, height))  # Helper dim. Makes linear transform easier.
  grid = np.stack((xv, yv, ones), axis=2)
  grid = torch.from_numpy(grid).float()
  return grid


class SpatialVAE(pl.LightningModule):

  def __init__(self,
               width,
               height,
               n_channels,
               n_hidden_units,
               n_hidden,
               n_unconstrained,
               has_rotation=True,
               has_translation=True,
               activation=nn.Tanh()):
    """
    Args:
    n_unconstrained: int
      The number of unconstrained latent variables. The encoder will output a
      mean and a logged standard deviation for each unconstrained latent
      variable.
    """
    super().__init__()

    self.width = width
    self.height = height
    self.n_channels = n_channels
    self.n_hidden_units = n_hidden_units
    self.n_hidden = n_hidden
    self.n_unconstrained = n_unconstrained
    self.has_rotation = has_rotation
    self.has_translation = has_translation
    self.activation = activation
    
    self.n_inputs_encoder = width * height * n_channels
    self.n_outputs_encoder = 2 * (n_unconstrained + has_rotation +
                                  2 * has_translation)
    self.n_inputs_decoder = n_unconstrained + 2

    self.grid_coords = make_normalized_grid(width, height)

    encoder_layers = []
    # Input layer.
    encoder_layers.append(nn.Linear(self.n_inputs_encoder, n_hidden_units))
    encoder_layers.append(self.activation)
    # Hidden layers.
    for _ in range(n_hidden):
      encoder_layers.append(nn.Linear(n_hidden_units, n_hidden_units))
      encoder_layers.append(self.activation)
    # Output layer.
    encoder_layers.append(nn.Linear(n_hidden_units, self.n_outputs_encoder))
    self.encoder = nn.Sequential(*encoder_layers)

    decoder_layers = []
    # Input layer.
    decoder_layers.append(nn.Linear(self.n_inputs_decoder, n_hidden_units))
    decoder_layers.append(self.activation)
    # Hidden layers.
    for _ in range(n_hidden):
      decoder_layers.append(nn.Linear(n_hidden_units, n_hidden_units))
      decoder_layers.append(self.activation)
    # Output layer.
    decoder_layers.append(nn.Linear(n_hidden_units, n_channels))
    decoder_layers.append(nn.Sigmoid())
    self.decoder = nn.Sequential(*decoder_layers)

  def encode(self, x):
    """Spatial-VAE encoder.

    Args:
    x: torch.tensor
      A tensor of shape [batch_size, n_channels * width * height]

    Returns:
    mu: torch.tensor
      A tensor of shape [batch_size, self.n_outputs_encoder / 2]. Means of
      uncostrained latent variables, theta (optional), and delta_x (optional)
    logvar: torch.tensor
      A tensor of shape [batch_size, self.n_outputs_encoder / 2]. Standard
      deviations of unconstrained latent variables, theta (optional), and
      delta_x (optional).
    """
    # encoded is of shape [batch_size, self.n_outputs_encoder]
    encoded = self.encoder(x)

    # split it into a tuple with 2 tensors of shape [batch_size,
    # self.n_outputs_encoder / 2]
    mu, logvar = encoded.split(self.n_outputs_encoder // 2, 1)

    return mu, logvar

  def sample(self, mu, logvar):
    """Sample from a normal distribution N(mu, exp(logvar))

    Args:
    mu: torch.tensor
      A tensor of shape [batch_size, self.n_outputs_encoder / 2]. Means of
      uncostrained latent variables, theta (optional), and delta_x (optional)
    logvar: torch.tensor
      A tensor of shape [batch_size, self.n_outputs_encoder / 2]. Log variance
      of uncostrained latent variables, theta (optional), and delta_x
      (optional).

    Returns:
    z: torch.tensor
      A tensor of shape [batch_size, self.n_outputs_encoder / 2]. unconstrained
      latent variables, theta (optional), and delta_x (optional).
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

  def decode(self, z):
    """Spatial-VAE decoder.

    Extracts theta and delta_x from z. Generates normalized grid coordinates
    and transforms them (rotation + translation). Concatenates the transformed
    coordinates with z, and inputs the resulting tensor to the decoder.

    Args:
    z: torch.tensor
      A tensor of shape [batch_size, self.n_outputs_encoder / 2]. unconstrained
      latent variables, theta (optional), and delta_x (optional).

    Returns:
    reconstruction: torch.tensor
      The reconstructed input. [batch_size, n_channels, width, height]
    """
    batch_size = z.shape[0]
    n_channels = self.n_channels
    width = self.width
    height = self.height

    # Transformation matrices applied to the normalized grid coordinates
    # in each batch. Shape [batch_size, n_cols=3, n_rows=3].
    transforms = torch.eye(3).repeat(batch_size, 1, 1).to(z.device)

    # Normalized grid coordinates that will be transformed and input to the
    # decoder.
    coords = (make_normalized_grid(self.width,
                                   self.height).repeat(batch_size, 1, 1,
                                                       1).to(z.device))

    if self.has_rotation:
      # Extract theta from z.
      theta = z[:, :1].view(batch_size)
      z = z[:, 1:]
      # Update the transformation matrices.
      transforms[:, 0, 0] = torch.cos(theta)
      transforms[:, 1, 0] = -torch.sin(theta)
      transforms[:, 0, 1] = torch.sin(theta)
      transforms[:, 1, 1] = torch.cos(theta)
    if self.has_translation:
      # Extract delta_x from z.
      delta_x = z[:, :2]
      z = z[:, 2:]
      # Update the transformation matrices.
      transforms[:, 2, :2] = delta_x

    # Transform the coordinates that go into the model. Coord shape will be
    # [batch_size, width, height, 2]
    transforms = transforms.view(batch_size, 1, 3, 3)
    coords = (coords @ transforms)[:, :, :, :2]  # Remove the helper dimensions.

    # Reshape the data, for the decoder (one coordinate at a time).
    coords = coords.view(batch_size * width * height, 2)
    # z has shape [batch_size, n_unconstrained]. It should become
    # [batch_size * width * height, n_unconstrained].
    z = z.repeat_interleave(width * height, 0)
    # Lastly, concatenate the transformed coordinates and the unconstrained
    # latent variables.
    # x has shape [batch_size * width * height, n_inputs_decoder]
    x = torch.cat((z, coords), -1)

    # Decode. [batch_size * width * height, n_channels]
    decoded = self.decoder(x)
    # Reshape. [batch_size, n_channels, width, height]
    decoded = decoded.view(batch_size, n_channels, width, height)

    return decoded

  def forward(self, x):
    """
    Args:
    x: torch.tensor
      A tensor of shape [batch_size, n_channels, width, height]

    Returns:
    reconstruction: torch.tensor
      The reconstructed input. [batch_size, n_channels, width, height]
    mu: torch.tensor
      A tensor of shape [batch_size, self.n_outputs_encoder / 2]. Means of
      uncostrained latent variables, theta (optional), and delta_x (optional)
    logvar: torch.tensor
      A tensor of shape [batch_size, self.n_outputs_encoder / 2]. Standard
      deviations of unconstrained latent variables, theta (optional), and
      delta_x (optional)
    """

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

  def loss(self, x, reconstruction, mu, logvar, sigma_theta):
    """Loss function.
    Sums the reconstruction loss with the individual KL divergences of
    the latent variables. The KL divergence is custom for rotation and
    translation transformations.

    Args:
    x: torch.tensor
      The original input [batch_size, n_channels, width, height]
    reconstruction: torch.tensor
      The reconstructed input. [batch_size, n_channels, width, height]
    mu: torch.tensor
      A tensor of shape [batch_size, self.n_outputs_encoder / 2]. Means of
      unconstrained latent variables, theta (optional), and delta_x (optional)
    logvar: torch.tensor
      A tensor of shape [batch_size, self.n_outputs_encoder / 2]. Standard
      deviations of unconstrained latent variables, theta (optional), and
      delta_x (optional).
    sigma: torch.tensor
      A single value tensor. Standard deviation of the prior on rotation.
      A good value would be pi as a large value result in an uniform prior.
    Returns:
    loss: torch.tensor
      The overall loss [overall_loss].
    """
    batch_size = x.shape[0]
    reconstruction_loss = F.binary_cross_entropy(
        reconstruction, x, reduction='sum') / batch_size

    kl_div = 0
    # Custom equation defined in Spatial VAE (Bepler et al) (2019)
    # Calculate KL Divergence for rotation variable
    # pseudo -0.5 - logvar + log_sigma + var/(2*sigma^2)
    if self.has_rotation:
      # Extract mu and logvar for theta from z.
      mu_theta, mu = mu[:, :1], mu[:, 1:]
      logvar_theta, logvar = logvar[:, :1], logvar[:, 1:]
      logstd_theta = 0.5 * logvar_theta

      kl_div_theta = (-0.5 - logstd_theta + torch.log(sigma_theta) +
                      (logvar_theta.exp() + mu_theta**2) /
                      (2 * sigma_theta**2)).sum(1)
      kl_div += kl_div_theta
      print(f'theta: {kl_div.mean()}')

    # Implementation based of Kingma and Welling (2014)
    # calculate KL Divergence for translation variables
    if self.has_translation:
      mu_delta_x, mu = mu[:, :2], mu[:, 2:]
      logvar_delta_x, logvar = logvar[:, :2], logvar[:, 2:]

      kl_div_delta_x = -0.5 * (1 + logvar_delta_x - mu_delta_x.pow(2) -
                               logvar_delta_x.exp()).sum(1)
      kl_div += kl_div_delta_x

    kl_div_z = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)
    kl_div += kl_div_z
    print(f'total: {kl_div.mean()}')

    elbo = reconstruction_loss + kl_div.mean()
    return elbo
