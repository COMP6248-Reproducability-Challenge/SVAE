# Enable autoreloading, when local modules are modified.
get_ipython().run_line_magic("load_ext", " autoreload")
get_ipython().run_line_magic("autoreload", " 2")


import os
import sys

# Allow absolute path imports.
project_root = os.path.abspath(os.path.join('../..'))
if project_root not in sys.path:
  sys.path.append(project_root)


import torch

from src.models.svae import SpatialVAE


torch.set_printoptions(precision=3, sci_mode=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Torch device: {device}')


batch_size = 8
n_channels = 3  # e.g RGB (set to 1 for MNIST).
width = 28  # Image width.
height = 28  # Image height.


# A dummy input.
x = torch.rand(batch_size, n_channels, width, height).to(device)


svae = SpatialVAE(
    width=width,
    height=height,
    n_channels=n_channels,
    n_hidden_units=500,
    n_layers=2,
    n_unconstrained=2).to(device)


reconstruction, mu, logstd = svae.forward(x)
reconstruction.shape
