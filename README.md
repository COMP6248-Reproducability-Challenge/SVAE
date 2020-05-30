# Reproducibility Challenge - Spatial-VAE

## Spatial-VAE.

The Spatial-VAE architecture, as defined in [1] is shown below.

<p align="center">
    <img src=".res/svae.svg">
</p>

Make sure to check our interactive demo of the model [here](https://comp6248-reproducability-challenge.github.io/SVAE/)! :innocent:

<p align="center">
    <img src=".res/interact.png">
</p>

## Getting Started

The experiments were run in `Python 3.7`. Start by installing PyTorch as per the [docs](https://pytorch.org/get-started/locally/). Then run the commands below (they are for Ubuntu and might be slightly different for other OS):

```bash
# Clone the repository.
git clone https://github.com/COMP6248-Reproducability-Challenge/SVAE.git
cd SVAE

# (Optional) Create a new Python environment and activate it.
python3 -m venv .env
source .env/bin/activate

# Install the dependencies.
pip install -r requirements.txt
```

## Train a Model

To train a model with the default parameters run:

```bash
python -m src.train
```

To see the available options, run:

```bash
python -m src.train --help
```

After the model is trained, a state dict (a `.pt` file) and the loss log (a `.csv` file) will be stored in `model_logs/`. The name of the files is `<dataset>_<svae><has_rotation><has_translation>_<n_unconstrained>`.

## Directory Structure

- data - the three MNIST datasets are here.
- doc - the report is here.
- model_logs - after training, models and logs will be stored here.
- src - the source code.
  - models - the PyTorch models.
    - `svae.py` - implementation of the SpatialVAE architecture.
    - `mnist_model.py` - adds training methods to SpatialVAE from `svae.py`.
  - notebooks - an notebook example of the model.
  - `train.py` - the training script.

## Other Branches

- report - contains the LaTeX source, and scripts for plotting the results; all logs are stored here.
- gh-pages - contains scripts for converting the PyTorch model to ONNX and the html files for the interactive demo.
- vvae - an implementation of a standard (vanilla) VAE, plus training scripts.

## References

[1] **Explicitly disentangling image content from translation and rotation with spatial-VAE** ([online](https://arxiv.org/abs/1909.11663))  
Tristan Bepler, Ellen D. Zhong, Kotaro Kelley, Edward Brignole, Bonnie Berger.
