import matplotlib.pyplot as plt
import numpy as np
import os
# plt.rcParams.update({'font.size': 4})


def main():
  scale = 2.7
  datasets = ['mnist', 'mnist_rotated', 'mnist_rotated_translated']
  dataset_names = ['MNIST', 'rotated MNIST', 'rotated & translated MNIST']
  zds = [2, 3, 5, 10]
  models = ['svae11', 'svae00', 'vvae']
  colors = ['C2', 'C1', 'C0']

  fig, axss = plt.subplots(figsize=(5.50107 * scale, 3.5 * scale),
                           nrows=3,
                           ncols=4,
                           sharex=True,
                           sharey=True)

  for r, (axs, dataset,
          dataset_name) in enumerate(zip(axss, datasets, dataset_names)):
    for c, (ax, zd) in enumerate(zip(axs, zds)):
      for model, color in zip(models, colors):
        filename = f'{dataset}_{model}_{zd}.csv'
        if filename in os.listdir('model_logs'):
          logs = np.genfromtxt(f'model_logs/{filename}',
                               delimiter=',',
                               skip_header=1)
          ax.plot(-logs[:, 0], c=color)
          ax.plot(-logs[:, 1], c=color, linestyle='--')

      ax.set_title(f'{dataset_name}, Z-D={zd}')
      ax.set_xticks([0, 50, 100, 150, 200])
      ax.set_yticks([-200, -180, -160, -140, -120, -100])
      ax.set_ylim([-200, -90])
      if r == 2:
        ax.set_xlabel('Epoch')
      if c == 0:
        ax.set_ylabel('ELBO')

  fig.tight_layout()
  fig.subplots_adjust(hspace=0.18,
                      wspace=0.04,)
  fig.savefig('plots/elbos.pdf')
  plt.show()


if __name__ == '__main__':
  main()