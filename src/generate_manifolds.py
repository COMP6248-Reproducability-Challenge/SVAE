import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from torchvision.utils import save_image
from models.svae import SpatialVAE


def make_manifold(model, filename, lim=2, extra_z=False):
  a = 2
  img = torch.empty((280, 280))
  grid_x = np.linspace(-a, a, 10)
  grid_y = np.linspace(-a, a, 10)[::-1]

  for i, yi in enumerate(grid_y):
    for j, xi in enumerate(grid_x):
      if extra_z:
        z_sample = torch.tensor([[0, 0, 0, xi, yi]]).float()
      else:
        z_sample = torch.tensor([[xi, yi]]).float()
      x_decoded = model.decode(z_sample).view(28, 28)
      img[28 * i:28 * (i + 1), 28 * j:28 * (j + 1)] = x_decoded

  # Invert
  img = -(img - 1).detach()
  # plt.imshow(img, cmap='gray')
  # plt.show()
  save_image(img, f'plots/{filename}_mf.png')


def main():
  datasets = ['mnist', 'mnist_rotated', 'mnist_rotated_translated']

  for dataset in datasets:
    model = SpatialVAE(width=28, height=28, n_channels=1, n_hidden_units=500, n_hidden=1, n_unconstrained=2, delta_x_prior=1.4, theta_prior=0.7853981634, has_rotation=True, has_translation=True)
    model.load_state_dict(torch.load(f'model_logs/{dataset}_svae11_2.pt', map_location=torch.device('cpu')))
    model.eval()
    make_manifold(model, f'{dataset}_svae11_2', 1, True)

  for dataset in datasets:
    model = SpatialVAE(width=28, height=28, n_channels=1, n_hidden_units=500, n_hidden=1, n_unconstrained=2, delta_x_prior=1.4, theta_prior=0.7853981634, has_rotation=False, has_translation=False)
    model.load_state_dict(torch.load(f'model_logs/{dataset}_svae00_2.pt', map_location=torch.device('cpu')))
    model.eval()
    make_manifold(model, f'{dataset}_svae00_2', 1)

if __name__ == '__main__':
  main()