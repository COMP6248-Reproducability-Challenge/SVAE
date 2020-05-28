import argparse
import math
import os

import pytorch_lightning as pl
import torch

from src.models.mnist_vanilla_model import MnistVanModel


def main():
  parser = argparse.ArgumentParser(description='Process some integers.')
  parser.add_argument(
      '--dataset',
      choices=['mnist', 'mnist_rotated', 'mnist_rotated_translated'],
      default='mnist',
      help='The dataset on which to train.')
  parser.add_argument('--n_hidden_units',
                      type=int,
                      default=500,
                      help='Number of hidden units.')
  parser.add_argument('--n_latent',
                      type=int,
                      default=2,
                      help='Number of latent variables.')
  parser.add_argument('--n_epochs',
                      type=int,
                      default=200,
                      help='Number of epochs to train for.')
  args = parser.parse_args()

  model = MnistVanModel(dataset=args.dataset,
                        width=28,
                        height=28,
                        n_channels=1,
                        n_hidden_units=args.n_hidden_units,
                        n_latent=args.n_latent)

  trainer = pl.Trainer(max_epochs=args.n_epochs,
                       checkpoint_callback=False,
                       logger=False,
                       early_stop_callback=False,
                       gpus=int(torch.cuda.is_available()))
  trainer.fit(model)

  # Save model weights and loss curves.
  if not os.path.exists('model_logs'):
    os.makedirs('model_logs')

  model_name = f'{args.dataset}_vvae_{args.n_latent}'
  torch.save(model.state_dict(), f'model_logs/{model_name}.pt')
  with open(f'model_logs/{model_name}.csv', 'w') as f:
    log = model.log
    f.write(','.join(log.keys()))
    f.write('\n')
    values = list(log.values())
    f.writelines(
        [f'{values[0][i]},{values[1][i]}\n' for i in range(len(values[0]))])


if __name__ == '__main__':
  main()