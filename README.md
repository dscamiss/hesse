# `hesse` 🧘‍♂️

![License](https://img.shields.io/badge/license-MIT-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3.9-blue.svg)
![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![Build](https://github.com/dscamiss/hesse/actions/workflows/python-package.yml/badge.svg)
[![codecov](https://codecov.io/gh/dscamiss/hesse/graph/badge.svg?token=Z3CGGZJ70B)](https://codecov.io/gh/dscamiss/hesse)

# Introduction

The goal of `hesse` is to provide simple functions that streamline the computation of Hessian
matrices and related quantities.

This can be inconvenient with the existing PyTorch functionality.
For example, suppose that we want to compute the Hessian matrix of a loss function 
with respect to model parameters.  To do so, we must

* Make a [functional version](https://pytorch.org/docs/stable/generated/torch.func.functional_call.html)
  of the loss function which is directly dependent on the model parameters, 
* Compute the Hessian data of the functional version, and
* Transform the Hessian data into the Hessian matrix.

The complexity increases if we want to

* Compute the Hessian matrix with respect to a proper subset of the model parameters,
* Compute a diagonal approximation of the Hessian matrix, or
* Accommodate batch data.

All of these operations are one-liners with `hesse`.  For example, to compute the Hessian matrix of

```python
  loss = loss_criterion(model(inputs), target)
```

with respect to the parameters of `model`, we can simply write

```python
  hessian_matrix = hesse.loss_hessian_matrix(model, loss_criterion, inputs, target)
```

# Installation

```bash
git clone https://github.com/dscamiss/hesse/
pip install -e hesse
```

# Usage

TODO

# TODO

- [ ] Add remaining test cases
- [ ] Add diagonal-only sharpness computation 
- [ ] Sparse storage where appropriate
- [ ] Test with multi-output model
- [ ] Test batch functions with batch data
