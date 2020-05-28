import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F


class SpatialVaeEncoder(nn.Module):

  def __init__(self):
    super().__init__()

    self.encoder = nn.Sequential(
      nn.Linear(784, 500),
      nn.Tanh(),
      nn.Linear(500, 500),
      nn.Tanh(),
      nn.Linear(500, 10),
    )

  def forward(self, x):
    x = x.reshape(280, 280, 4)
    x = torch.narrow(x, dim=2, start=3, length=1)
    x = x.reshape(1, 1, 280, 280)
    x = F.avg_pool2d(x, 10, stride=10, ceil_mode=False)
    x = x.view(-1, 28 * 28)
    x = x / 255.0
    return self.encoder(x.view(1, 784))


class SpatialVaeDecoder(nn.Module):

  def __init__(self):
    super().__init__()

    self.decoder = nn.Sequential(
      nn.Linear(4, 500),
      nn.Tanh(),
      nn.Linear(500, 500),
      nn.Tanh(),
      nn.Linear(500, 1),
      nn.Sigmoid(),
    )

  def forward(self, x):
    x = self.decoder(x)
    x = - (x - 1)
    x = x * 255
    x = x.view(28, 28)
    return x