# Reproducibility Challenge - Stacked Capsule Autoencoders

## Getting Started

The experiments were run in `Python 3.7`.

```powershell
# Clone the repository.
git clone https://github.com/yannidd/reproducibility.git

# (Optional) Create a new Python environment and activate it.
# Windows:
python -m venv .env
.env\Scripts\activate.bat
# Unix or MacOS:
python3 -m venv .env
source .env/bin/activate

# Install the dependencies.
# Windows:
pip install -r requirements.txt
# Unix or MacOS:
pip3 install -r requirements.txt
```

## Useful Repositories

- Official Stacked Capsule Autoencoders repository - [github](https://github.com/google-research/google-research/tree/master/stacked_capsule_autoencoders)
- Official set transformer implementation - [github](https://github.com/juho-lee/set_transformer)

## Useful Theory

A quick recap on some useful theory that is mentioned in the paper.

### Similarity Transform

A similarity transform allows for 4 DoF - translation + rotation + scale [6].  

<p align="center">
    <img src=".res/similarity_transform_cats.jpg" width="512">
</p>

Change `t_x` and `t_y` for translation, `a` for scaling and `theta` for rotation.

<p align="center">
    <img src=".res/eq_similarity_transform.svg">
</p>

### Set Transformer

Extremely briefly: A set transformer [7] takes as an input a variable length set of tensors, and transforms it to a constant shape tensor. The output is permutation invariant.

The toy example provided in the paper might help in understanding what a set transformer does. The example follows: Given a set with an arbitrary length of real numbers `{x_1, ..., x_n}`, return `max({x_1, ..., x_n})`.

## Further Reading

A good start to understand Capsule Networks is [4] - a series of 4 blog posts. If the Stacked Capsule Autoencoder paper [1] seems too technical, [2] provides a high level overview of the approach. Knowledge of set transformers [7] is required to understand the paper.

## References

[1] **Stacked Capsule Autoencoders Paper.** ([online](https://arxiv.org/abs/1906.06818))  
Kosiorek, Adam Roman and Sabour, Sara and Teh, Yee Whye and Hinton, Geoffrey Everest, 2019. Advances in Neural Information Processing Systems.

[2] **Stacked Capsule Autoencoders Blog Post.** ([online](http://akosiorek.github.io/ml/2019/06/23/stacked_capsule_autoencoders.html))  
Adam Kosiorek, 2019.

[3] **Stacked Capsule Autoencoders GitHub Repository.** ([online](https://github.com/google-research/google-research/tree/master/stacked_capsule_autoencoders))

[4] **Understanding Hinton’s Capsule Networks. Part I: Intuition.** ([online](https://medium.com/ai%C2%B3-theory-practice-business/understanding-hintons-capsule-networks-part-i-intuition-b4b559d1159b))  
Max Pechyonkin, 2017. AI³ | Theory, Practice, Business.

[5] **Awesome Capsule Networks.** ([online](https://github.com/sekwiatkowski/awesome-capsule-networks))  

[6] **Geometric Transformations** ([online](https://courses.cs.washington.edu/courses/csep576/11sp/pdf/Transformations.pdf))  
Larry Zitnick.

[7] **Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks** ([online](https://arxiv.org/pdf/1810.00825.pdf))  
Juho Lee, Yoonho Lee, Jungtaek Kim, Adam R. Kosiorek, Seungjin Choi, Yee Whye Teh, 2019.
