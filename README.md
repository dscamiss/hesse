# `hesse` 🧘‍♂️

![License](https://img.shields.io/badge/license-MIT-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3.9-blue.svg)
![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![Build](https://github.com/dscamiss/hesse/actions/workflows/python-package.yml/badge.svg)
[![codecov](https://codecov.io/gh/dscamiss/hesse/graph/badge.svg?token=Z3CGGZJ70B)](https://codecov.io/gh/dscamiss/hesse)

# Introduction

Computing Hessians and Hessian-related quantities in PyTorch is inconvenient.  To illustrate this,
suppose that we want to compute the sharpness (that is, the largest absolute eigenvalue) of a loss function `loss` with 
respect to the parameters of `model`.  To do so, we have to 

* Construct a "functional version" of `loss` which is functionally dependent on the parameters of `model`, 
* Compute the Hessian of the functional version,
* Transform the resulting Hessian data into an actual matrix, and
* Compute its largest absolute eigenvalue.

More work is needed if we are only interested in the Hessian with respect to a subset of the parameters of `model`, 
if we are interested in using diagonal approximations of Hessians, if batch data is involved, and so on.

The goal of `hesse` is to provide simple functions that streamline the computation of Hessians and Hessian-related 
quantities.  For example, the sharpness computation reduces to

```python
  sharpness = loss_sharpness(model, loss_criterion, inputs, target)
```

# Installation

TODO

# Usage

TODO
