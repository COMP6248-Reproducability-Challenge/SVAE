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
    print(self.encoder(x.view(1, 784)))
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
    x = x * 255
    x = x.view(28, 28)
    return x