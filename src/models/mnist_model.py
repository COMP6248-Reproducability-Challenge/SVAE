import math
import os
import tarfile

import torch
from torch.utils.data import DataLoader

from src.models.svae import SpatialVAE


class MnistModel(SpatialVAE):

  def __init__(self,
               dataset='mnist',
               sigma_theta=torch.tensor(math.pi),
               **kwargs):
    super().__init__(**kwargs)
    if dataset in {'mnist', 'mnist_rotated', 'mnist_rotated_translated'}:
      self.dataset = dataset
    else:
      raise KeyError('Dataset should be one of mnist, mnist_rotated, '
                     'mnist_rotated_translated')
    self.sigma_theta = sigma_theta
    self.log = {'training': [], 'test': []}

  def prepare_data(self):
    if self.dataset not in os.listdir('data'):
      tar = tarfile.open(f'data/{self.dataset}.tar.xz', 'r:xz')
      tar.extractall('data')
      tar.close()

    self.training_data = torch.load(f'data/{self.dataset}/training.pt') / 255.0
    self.test_data = torch.load(f'data/{self.dataset}/test.pt') / 255.0

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=0.0001)

  def train_dataloader(self):
    return DataLoader(self.training_data, num_workers=4, batch_size=100)

  def val_dataloader(self):
    return DataLoader(self.test_data, num_workers=4, batch_size=100)

  def training_step(self, batch, batch_idx):
    batch_size, width, height = batch.shape
    batch = batch.view(batch_size, 1, width, height)
    reconstruction, mu, logvar = self.forward(batch)
    loss = self.loss(batch, reconstruction, mu, logvar, self.sigma_theta)
    return {'loss': loss}

  def validation_step(self, batch, batch_idx):
    batch_size, width, height = batch.shape
    batch = batch.view(batch_size, 1, width, height)
    reconstruction, mu, logvar = self.forward(batch)
    loss = self.loss(batch, reconstruction, mu, logvar, self.sigma_theta)
    return {'val_loss': loss}

  def training_epoch_end(self, outputs):
    loss = sum([output['loss'] for output in outputs]) / len(self.training_data)
    self.log['training'].append(loss.item())
    return {'loss': loss}

  def validation_epoch_end(self, outputs):
    loss = sum([output['val_loss'] for output in outputs]) / len(self.test_data)
    self.log['test'].append(loss.item())
    return {'val_loss': loss}

