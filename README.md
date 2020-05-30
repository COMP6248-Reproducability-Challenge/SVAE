# Reproducibility Challenge - Spatial-VAE

This branch contains an implementation of a vanilla VAE (VVAE).

## Train a Model

To train a model with the default parameters run:

```bash
python -m src.train
```

To see the available options, run:

```bash
python -m src.train --help
```

After the model is trained, a state dict (a `.pt` file) and the loss log (a `.csv` file) will be stored in `model_logs/`. The name of the files is `<dataset>_<vvae>_<n_unconstrained>`.