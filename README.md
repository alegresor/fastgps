# `FastGPs`: Fast Gaussian Process Regression in Python

[![Docs](https://github.com/alegresor/fastgps/actions/workflows/docs.yml/badge.svg?branch=main)](https://alegresor.github.io/fastgps/)
[![Tests](https://github.com/alegresor/fastgps/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/alegresor/fastgps/actions/workflows/tests.yml)

Gaussian process regression (GPR) on $n$ data points typically costs $\mathcal{O}(n^3)$ computations and $\mathcal{O}(n^2)$ storage. Fast GPR only costs $\mathcal{O}(n \log n)$ computations and $\mathcal{O}(n)$ storage by forcing nice structure into the $n \times n$ Gram matrix of pairwise kernel evaluations. Fast GPR requires

1. control over the design of experiments, i.e., sampling at fixed locations, and
2. Using special kernel forms that are practically performant but generally uncommon, e.g., one *cannot* use common kernels such as the Squared Exponential, Matern, or Rational Quadratic.

## Installation

```bash
pip install fastgps
```

## Resources

The [FastGPs documentation](https://alegresor.github.io/fastgps/) contains a detailed **package reference** documenting classes including thorough doctests. A number of **example notebooks** are also rendered into the documentation from `fastgps/docs/examples/`. We recommend reading [Aleksei Sorokin's slides on Fast GPR](https://github.com/alegresor/alegresor.github.io/blob/main/presentations/2025_FastGPs_MCM.pdf) which he presented at [MCM 2025 Chicago](https://fjhickernell.github.io/mcm2025/).

## Fast GPR Methods

We currently support two flavors of fast GPR:

1. Pairing integration lattice point sets with shift-invariant (SI) kernels which creates circulant Gram matrices that are diagonalizable by Fast Fourier Transforms (FFTs). SI kernels are periodic and arbitrarily smooth.
2. Pairing digital nets (e.g. Sobol' point sets) with digitally-shift-invariant (DSI) kernels which creates Gram matrices diagonalizable by Fast Walsh Hadamard Transforms (FWHTs). DSI kernels are discontinuous, yet versions exist for which the corresponding Reproducing Kernel Hilbert Space (RKHSs) contains arbitrarily smooth functions.

## Software Features

A reference standard GP implementation is available alongside the fast GPR implementations. All GPR methods support:

- **GPU computations** as `FastGPs` is built on the `PyTorch` stack.
- **Batching** of both outputs (for functions with tensor outputs) and parameters (with flexibly shareable parameters among batched outputs).
- **Multi-Task GPs** with product kernels and generalized fast multi-task GPR.
- **Derivative Information** of arbitrarily high order.
- **Bayesian Cubature** for approximating integrals or expectations.
- **Flexible kernel parameterizations** from the [`QMCPy` package](https://qmcsoftware.github.io/QMCSoftware/).
- **Efficient variance projections** for determining if and where to sample next.

## References

This package is based off of the following publications

1. Jagadeeswaran, Rathinavel, and Fred J. Hickernell. "Fast automatic Bayesian cubature using lattice sampling." Statistics and Computing 29.6 (2019): 1215-1229.

2. Jagadeeswaran, Rathinavel, and Fred J. Hickernell. "Fast automatic Bayesian cubature using Sobol’ sampling." Advances in Modeling and Simulation: Festschrift for Pierre L'Ecuyer. Cham: Springer International Publishing, 2022. 301-318.

3. Rathinavel, Jagadeeswaran. Fast automatic Bayesian cubature using matching kernels and designs. Illinois Institute of Technology, 2019.
