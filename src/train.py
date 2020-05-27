import pytorch_lightning as pl
import torch

from src.models.mnist_model import MnistModel


def main():
  model = MnistModel(dataset='mnist',
                     width=28,
                     height=28,
                     n_channels=1,
                     n_hidden_units=500,
                     n_hidden=1,
                     n_unconstrained=2,
                     has_rotation=True,
                     has_translation=True)

  trainer = pl.Trainer(
      train_percent_check=1,
      val_percent_check=1,
      max_epochs=4,
  )  # pt.trainer(gpus=1) in order to use GPU instead of CPU
  trainer.fit(model)


if __name__ == '__main__':
  main()
