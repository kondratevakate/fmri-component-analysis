# fmri-component-analysis

Table of contents:

DL methods for Blind Source Separation:
https://github.com/sigsep/open-unmix-pytorch One of the winners of MUSDB18 signal separation competition

https://youtu.be/Xr7UOWIniCM

Open-Unmix is based on a three-layer bidirectional deep LSTM. The model learns to predict the magnitude spectrogram of a target, like vocals, from the magnitude spectrogram of a mixture input.

ANICA. Adversarial Non-linear ICA:
http://github.com/anica Maximizing Independence with GANs for Non-linear ICA. Implemented in 2017, for Python2 and Tensorflow1

ICE-BeeM: Identifiable Conditional Energy-Based Deep Models Based on Nonlinear ICA. https://github.com/ilkhem/icebeem
Under certain constraints, non-linearly mixed sources are proven to be uniquely identifiable:

Nonstationarity (time-contrastive learning) https://arxiv.org/abs/1605.06336
Temporal dependencies (permutation-contrastive learning) https://arxiv.org/pdf/1805.08651.pdf
Existence of auxiliary variable (e.g. iVAE) https://arxiv.org/abs/1907.04809
Vanila VAE for 2d slices of task-based fMRI. VAE_for_fMRI folder in current repository

3D VAE regression network for MRI. Published in MICCAI 2019

https://github.com/QingyuZhao/VAE-for-Regression
