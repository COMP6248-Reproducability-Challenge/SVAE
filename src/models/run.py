import os
from argparse import Namespace

from src.models.pl_model import Model

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.logging import TensorBoardLogger

os.chdir('../../')


def main(specs):
  model = Model(specs.dataset, specs.width, specs.height, specs.n_channels,
                specs.n_hidden_units, specs.n_layers, specs.n_unconstrained)
  checkpoint_callback = ModelCheckpoint(filepath='src/models/log/')
  logger = TensorBoardLogger("src/models/log/tb_logs",
                             name="spatial_vae_report")
  trainer = pl.Trainer(
    train_percent_check=0.05,
    max_epochs=100,
    checkpoint_callback=checkpoint_callback,
    logger=logger)  # Trainer(gpus=1) in order to use GPU instead of CPU
  trainer.fit(model)


if __name__ == '__main__':
  params = dict(dataset='MNIST',
                width=28,
                height=28,
                n_channels=1,
                n_hidden_units=3,
                n_layers=4,
                n_unconstrained=2)
  h_params = Namespace(**params)
  main(h_params)
