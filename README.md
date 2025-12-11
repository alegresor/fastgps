# `fastgps`: Fast Gaussian Process Regression in Python

[![Docs](https://github.com/alegresor/fastgps/actions/workflows/docs.yml/badge.svg?branch=main)](https://alegresor.github.io/fastgps/)
[![Tests](https://github.com/alegresor/fastgps/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/alegresor/fastgps/actions/workflows/tests.yml)

## Installation

```bash
pip install fastgps
```

## Overview

Gaussian process (GP) regression on $n$ data points typically require $\mathcal{O}(n^3)$ computations and $\mathcal{O}(n^2)$ storage. Fast GPs only require $\mathcal{O}(n \log n)$ computations and $\mathcal{O}(n)$ storage by forcing nice structure into the $n \times n$ Gram matrix of pairwise kernel evaluations. Fast GPs require

- Control over the design of experiments, i.e., sampling at fixed locations which we will choose to be **quasi-random (low-discrepancy) sequences**, and
- Using special kernel forms that are practically performant but generally uncommon, e.g., one *cannot* use common kernels such as the Squared Exponential, Matern, or Rational Quadratic. We will use **(digitally)-shift invariant kernels**.

## Scope

`fastgps` currently support two flavors:

1. Pairing **rank-1 integration lattices** with **shift-invariant (SI) kernels** creates circulant Gram matrices that are diagonalizable by Fast Fourier Transforms (**FFTs**). SI kernels are periodic and arbitrarily smooth.
2. Pairing **digital sequences** (e.g. Sobol' sequences) with **digitally-shift-invariant (DSI) kernels** creates Gram matrices diagonalizable by Fast Walsh-Hadamard Transforms (**FWHTs**). DSI kernels are discontinuous, yet versions exist for which the corresponding Reproducing Kernel Hilbert Space (RKHSs) contains arbitrarily smooth functions.

## Features

A reference standard GP implementation is available alongside the fast GP implementations. All GP methods support:

- **GPU computations** as `fastgps` is built on the `PyTorch` stack.
- **Batching** of both outputs (for functions with tensor outputs) and parameters (with flexibly shareable parameters among batched outputs).
- **Multi-Task GPs** with product kernels and generalized fast multi-task GPs.
- **Derivative Information** of arbitrarily high order.
- **Bayesian Cubature** for approximating integrals or expectations.
- **Flexible kernel parameterizations** from the [`QMCPy` package](https://qmcsoftware.github.io/QMCSoftware/).
- **Efficient variance projections** for determining if and where to sample next.

## Resources

The [`fastgps` documentation](https://alegresor.github.io/fastgps/) contains a detailed **package reference** documenting classes including thorough doctests. A number of **example notebooks** are also rendered into the documentation from `fastgps/docs/examples/`. We recommend reading [Aleksei Sorokin's slides on Fast GPs](https://github.com/alegresor/alegresor.github.io/blob/main/presentations/2025_FastGPs_MCM.pdf) which he presented at [MCM 2025 Chicago](https://fjhickernell.github.io/mcm2025/).

## Citation

If you find the `fastgps` package helpful in your work, please consider citing the following papers

```bibtex
@phdthesis{sorokin.thesis,
  title               = {Algorithms and scientific software for quasi-{M}onte {C}arlo, fast {G}aussian process regression, and scientific machine learning},
  author              = {Aleksei G. Sorokin},
  year                = {2025},
  school              = {Illinois Institute of Technology},
  journal             = {ArXiv preprint},
  volume              = {abs/2511.21915},
  url                 = {https://arxiv.org/abs/2511.21915},
}

@inproceedings{sorokin.fastgps_probnum25,
  title               = {Fast {G}aussian process regression for high dimensional functions with derivative information},
  author              = {Sorokin, Aleksei G. and Robbe, Pieterjan and Hickernell, Fred J.},
  year                = {2025},
  booktitle           = {Proceedings of the First International Conference on Probabilistic Numerics},
  publisher           = {{PMLR}},
  series              = {Proceedings of Machine Learning Research},
  volume              = {271},
  pages               = {35--49},
  url                 = {https://proceedings.mlr.press/v271/sorokin25a.html},
  editor              = {Kanagawa, Motonobu and Cockayne, Jon and Gessner, Alexandra and Hennig, Philipp},
  pdf                 = {https://raw.githubusercontent.com/mlresearch/v271/main/assets/sorokin25a/sorokin25a.pdf},
}

@article{sorokin.FastBayesianMLQMC,
  title               = {Fast {B}ayesian multilevel quasi-{M}onte {C}arlo},
  author              = {Aleksei G. Sorokin and Pieterjan Robbe and Gianluca  Geraci and Michael S. Eldred and Fred J. Hickernell},
  year                = {2025},
  journal             = {ArXiv preprint},
  volume              = {abs/2510.24604},
  url                 = {https://arxiv.org/abs/2510.24604},
}
```
