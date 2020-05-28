import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
def main():
  img_namess = [
    ['mnist_samples.png', 'mnist_vvae_2_mf.png', 'mnist_svae00_2_mf.png', 'mnist_svae11_2_mf.png'],
    ['mnist_rotated_samples.png', 'mnist_rotated_vvae_2_mf.png', 'mnist_rotated_svae00_2_mf.png', 'mnist_rotated_svae11_2_mf.png'],
    ['mnist_rotated_translated_samples.png', 'mnist_rotated_translated_vvae_2_mf.png', 'mnist_rotated_translated_svae00_2_mf.png', 'mnist_rotated_translated_svae11_2_mf.png'],
  ]
  top_titles = iter(['Example images', 'vanilla-VAE', r'spatial-VAE $(\theta=0, \Delta x=0)$', 'spatial-VAE'])
  side_labels = iter(['MNIST', 'rotated MNIST', 'rotated & translated MNIST'])
  scale = 2.7
  fig, axss = plt.subplots(figsize=(5.50107 * scale, 4.1258025 * scale),
                           nrows=3,
                           ncols=4,
                           sharex=True,
                           sharey=True)

  for r, (axs, img_names) in enumerate(zip(axss, img_namess)):
    for c, (ax, img_name) in enumerate(zip(axs, img_names)):
      ax.imshow(mpimg.imread(f'plots/{img_name}'))
      ax.set_xticks([])
      ax.set_yticks([])
      if r == 0:
        ax.set_title(next(top_titles))
      if c == 0:
        ax.set_ylabel(next(side_labels))
        ax.hlines(np.arange(28, 280, 28), 0, 280, linestyle='--', linewidth=1)
        ax.vlines(np.arange(28, 280, 28), 0, 280, linestyle='--', linewidth=1)
  
  fig.tight_layout()
  fig.subplots_adjust(hspace=0.06, wspace=0.0)
  fig.savefig('plots/manifolds.pdf')
  plt.show()



if __name__ == "__main__":
    main()