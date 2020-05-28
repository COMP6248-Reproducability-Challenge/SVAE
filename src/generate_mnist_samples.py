import matplotlib.pyplot as plt
import torch
import numpy as np
import os


def main():
  training = torch.load('data/mnist/training.pt') 
  idxs = np.random.choice(training.shape[0], 100, replace=False)
  training = training[idxs]

  img = np.empty((280, 280))

  for r in range(10):
    for c in range(10):
      pass

if __name__ == '__main__':
  main()