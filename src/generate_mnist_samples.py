import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from torchvision.utils import save_image

def main():
  datasets = ['mnist', 'mnist_rotated', 'mnist_rotated_translated']

  for dataset in datasets:
    training = torch.load(f'data/{dataset}/training.pt') / 255.0
    idxs = np.random.choice(training.shape[0], 100, replace=False)
    idxs = iter(idxs)

    img = torch.empty((280, 280))

    for i in range(10):
      for j in range(10):
        img[28 * i:28 * (i + 1), 28 * j:28 * (j + 1)] = training[next(idxs)]

    # Invert
    img = - (img - 1)

    save_image(img, f'plots/{dataset}_samples.png')

if __name__ == '__main__':
  main()