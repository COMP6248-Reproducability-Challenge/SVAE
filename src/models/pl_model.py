from collections import OrderedDict

import numpy as np
from src.models.svae import SpatialVAE

import torch
import torchvision
from livelossplot import PlotLosses
from torch.optim import Adam
from torch.utils.data import DataLoader


class Model(SpatialVAE):

  def __init__(self,
               dataset,
               width,
               height,
               n_channels,
               n_hidden_units,
               n_layers,
               n_unconstrained,
               has_rotation=True,
               has_translation=True,
               activation='tanh'):
    super().__init__(width,
                     height,
                     n_channels,
                     n_hidden_units,
                     n_layers,
                     n_unconstrained,
                     has_rotation=True,
                     has_translation=True,
                     activation='tanh')
    self.dataset = dataset
    self.pi = torch.tensor(np.pi).float().unsqueeze(0)
    self.epoch = 0
    groups = {'Real Time Loss Plot': ['Training Loss', 'Validation Loss']}
    self.liveloss = PlotLosses(groups=groups)
    self.logs = {}

  def prepare_data(self):
    """
    Loading the desired dataset as prompted by the user
    """
    if self.dataset == 'MNIST':
      transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
      mnist_train = torchvision.datasets.MNIST('../../src/data/mnist/',
                                                train=True,
                                                download=True,
                                                transform=transform).data.float()
      mnist_val = torchvision.datasets.MNIST('../../src/data/mnist/',
                                              train=False,
                                              download=True,
                                              transform=transform).data.float()
      self.train_dataset = mnist_train
      self.val_dataset = mnist_val
    elif self.dataset == 'MNIST_Rotated':
      mnist_train = torch.Tensor(np.load('../../src/data/mnist_rotated/images_train.npy'))
      mnist_val = torch.Tensor(np.load('../../src/data/mnist_rotated/images_test.npy'))
      self.train_dataset = mnist_train
      self.val_dataset = mnist_val
    elif self.dataset == 'MNIST_Translated':
      mnist_train = torch.Tensor(np.load('../../src/data/mnist_rotated_translated/images_train.npy'))
      mnist_val = torch.Tensor(np.load('../../src/data/mnist_rotated_translated/images_test.npy'))
      self.train_dataset = mnist_train
      self.val_dataset = mnist_val
    elif self.dataset == 'Galaxy_Zoo':
      mnist_train = torch.Tensor(np.load('../../src/data/galaxy_zoo/galaxy_zoo_train.npy'))
      mnist_val = torch.Tensor(np.load('../../src/data/galaxy_zoo/galaxy_zoo_test.npy'))
      self.train_dataset = mnist_train
      self.val_dataset = mnist_val
    else:
      print("Please choose between the available datasets: MNIST, MNIST_Rotated, MNIST_Translated and Galaxy_Zoo")

  def configure_optimizers(self):
    return Adam(self.parameters(), lr=1e-3)

  def train_dataloader(self):
    return DataLoader(self.train_dataset, batch_size=100)

  def val_dataloader(self):
    return DataLoader(self.val_dataset, batch_size=100)

  def training_step(self, batch, batch_idx):
    """
    Training loop definition.

    When using the Galaxy_Zoo dataset, the batch size shape has
    been accordingly reshaped in order to take into account for the
    3 input channels (instead of one as with the MNIST datasets.
    """
    logs = {}
    if self.dataset == 'Galaxy_Zoo':
      x = batch.view(batch.shape[0], 3, batch.shape[1], batch.shape[2])
    else:
      x = batch.view(batch.shape[0], 1, batch.shape[1], batch.shape[2])
    reconstruction, mu, logstd = self.forward(x)
    loss_train = self.loss(x, reconstruction, mu, logstd, self.pi)
    # adding logging
    tqdm_dict = {'train_loss': loss_train}
    output = OrderedDict({
      'loss': loss_train,
      'progress_bar': tqdm_dict
    })
    return output

  def validation_step(self, batch, batch_idx):
    """
    Validation loop definition.

    When using the Galaxy_Zoo dataset, the batch size shape has
    been accordingly reshaped in order to take into account for the
    3 input channels (instead of one as with the MNIST datasets.
    """
    if self.dataset == 'Galaxy_Zoo':
      x = batch.view(batch.shape[0], 3, batch.shape[1], batch.shape[2])
    else:
      x = batch.view(batch.shape[0], 1, batch.shape[1], batch.shape[2])
    reconstruction, mu, logstd = self.forward(x)
    loss_val = self.loss(x, reconstruction, mu, logstd, self.pi)
    tqdm_dict = {'val_loss': loss_val}
    output = OrderedDict({
      'val_loss': loss_val,
      'progress_bar': tqdm_dict
    })
    return output

  def training_epoch_end(self, outputs):
    train_loss_mean = 0
    for output in outputs:
      train_loss = output['loss']
      if self.trainer.use_dp or self.trainer.use_ddp2:
        train_loss = torch.mean(train_loss)
      train_loss_mean += train_loss

    train_loss_mean /= len(outputs)
    result = {
      'end_train_loss': train_loss_mean,
      'step': self.epoch
    }
    self.logs['Training Loss'] = train_loss_mean.item()
    return {'log': result}

  def validation_epoch_end(self, outputs):
    val_loss_mean = 0
    for output in outputs:
      val_loss = output['val_loss']
      if self.trainer.use_dp or self.trainer.use_ddp2:
        val_loss = torch.mean(val_loss)
      val_loss_mean += val_loss

    val_loss_mean /= len(outputs)
    result = {
      'end_val_loss': val_loss_mean,
      'step': self.epoch
    }
    self.logs['Validation Loss'] = val_loss_mean.item()
    return {'log': result}

  def on_epoch_end(self):
    self.epoch += 1
    self.liveloss.update(self.logs)
    self.liveloss.send()
