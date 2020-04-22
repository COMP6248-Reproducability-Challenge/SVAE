import os
import sys
import torch
import torchvision
from argparse import Namespace
import pytorch_lightning as pl
from src.models.pl_model import Model


def main(specs):
    model = Model(specs.dataset, specs.width, specs.height, specs.n_channels,
                  specs.n_hidden_units, specs.n_layers, specs.n_unconstrained)
    trainer = pl.Trainer()  # Trainer(gpus=1) in order to use GPU instead of CPU
    trainer.fit(model)


if __name__ == '__main__':
    params = dict(dataset ='MNIST',
                  width=28,
                  height=28,
                  n_channels=1,
                  n_hidden_units=3,
                  n_layers=4,
                  n_unconstrained=2)
    h_params = Namespace(**params)
    main(h_params)
