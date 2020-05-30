import math
import os
import tarfile

import torch
from torch.utils.data import DataLoader

from src.models.svae import SpatialVAE


class MnistModel(SpatialVAE):

  def __init__(self,
               dataset='mnist',
               **kwargs):
    """See SpatialVAE for the other arguments.
    Args:
      dataset: string
        One of 'mnist', 'mnist_rotated', 'mnist_rotated_translated'.
    """
    super().__init__(**kwargs)
    if dataset in {'mnist', 'mnist_rotated', 'mnist_rotated_translated'}:
      self.dataset = dataset
    else:
      raise KeyError('Dataset should be one of mnist, mnist_rotated, '
                     'mnist_rotated_translated')
    self.log = {'training': [], 'test': []}
    self.should_log = False

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
    loss = self.loss(batch, reconstruction, mu, logvar)
    return {'loss': loss, 'running_loss': loss.item() * batch_size}

  def validation_step(self, batch, batch_idx):
    batch_size, width, height = batch.shape
    batch = batch.view(batch_size, 1, width, height)
    reconstruction, mu, logvar = self.forward(batch)
    loss = self.loss(batch, reconstruction, mu, logvar)
    return {'val_loss': loss, 'running_loss': loss.item() * batch_size}

  def training_epoch_end(self, outputs):
    loss = sum([output['running_loss'] for output in outputs]) / len(
        self.training_data)
    if self.should_log:
      self.log['training'].append(loss)
    return {'loss': loss}

  def validation_epoch_end(self, outputs):
    loss = sum([output['running_loss'] for output in outputs]) / len(
        self.test_data)
    if self.should_log:
      self.log['test'].append(loss)
    return {'val_loss': loss}

  def on_train_start(self):
    self.should_log = True

  def on_train_end(self):
    self.should_log = False
