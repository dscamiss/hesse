# `hesse` 🧘‍♂️

![License](https://img.shields.io/badge/license-MIT-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3.9-blue.svg)
![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![Build](https://github.com/dscamiss/hesse/actions/workflows/python-package.yml/badge.svg)
[![codecov](https://codecov.io/gh/dscamiss/hesse/graph/badge.svg?token=Z3CGGZJ70B)](https://codecov.io/gh/dscamiss/hesse)

# Introduction

The goal of `hesse` is to simplify the computation of Hessians (and related quantities) in PyTorch.  

In particular, the goal is to simplify the computation of:

* **Model Hessians** (these are the Hessians of a given model with respect to its trainable parameters);
* **Loss function Hessians** (these are the Hessians of `loss(model(inputs), target)` with respect to `model`'s trainable parameters).

This is achieved with user-friendly wrappers for `torch.func` transforms.

# Installation

In an existing Python 3.9+ environment:

```bash
git clone https://github.com/dscamiss/hesse/
pip install ./hesse
```

# Example

## Setup

Create a toy multi-input, multi-output model.

```python
import torch
from torch import Tensor

class MimoModel(torch.nn.Module):
    """Multi-input, multi-output model."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.A = torch.nn.Parameter(torch.randn(input_dim, output_dim))
        self.B = torch.nn.Parameter(torch.randn(input_dim, output_dim))

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Run forward pass.

        Args:
            x: First input tensor of shape (b, n).
            y: Second input tensor of shape (b, n).

        Returns:
            The matrix

            [ tr(A^t A) x_{    0, :}   tr(B^t B) y_{    0, :} ]
            [ tr(A^t A) x_{    1, :}   tr(B^t B) y_{    1, :} ]
            [           :                        :            ]
            [ tr(A^t A) x_{b - 1, :}   tr(B^t B) y_{b - 1, :} ].
        """
        rows_1 = torch.trace(self.A.T @ self.A) * x
        rows_2 = torch.trace(self.B.T @ self.B) * y
        return torch.hstack((rows_1, rows_2))
```

Make an instance of `MimoModel` along with some batch inputs.

```python
input_dim = 2
output_dim = 2
model = MimoModel(input_dim, output_dim)

x = torch.Tensor(
    [
        [1.0, 2.0],
        [3.0, 4.0],
    ]
)
y = -1.0 * x
```

## Full Hessian matrix

Computing the full Hessian matrix of `model` is easy:

```python
hessian = hesse.model_hessian_matrix(model=model, inputs=(x, y))
```

We can now verify the correctness of the result.

Note that in general, the shape of the Hessian matrix will be `(batch_size, output_size, ...)`.  In this instance, `batch_size = 2` and `output_size = 2`.

```python
expected = torch.zeros([2, 4, 8, 8]))

expected[0][0][:4, :4] =  2.0 * torch.eye(4)
expected[0][1][:4, :4] =  4.0 * torch.eye(4)
expected[0][2][4:, 4:] = -2.0 * torch.eye(4)
expected[0][3][4:, 4:] = -4.0 * torch.eye(4)
expected[1][0][:4, :4] =  6.0 * torch.eye(4)
expected[1][1][:4, :4] =  8.0 * torch.eye(4)
expected[1][2][4:, 4:] = -6.0 * torch.eye(4)
expected[1][3][4:, 4:] = -8.0 * torch.eye(4)

assert hessian.equal(expected), "Error in Hessian values"
```

## Reduced Hessian matrix

Computing the Hessian matrix of `model` with respect to a subset of the model parameters is also easy:

```python
hessian = hesse.model_hessian_matrix(model=model, inputs=(x, y), params=("A"))
```

We can again verify the correctness of the result.

```python
expected = torch.zeros([2, 4, 4, 4])

expected[0][0] = 2.0 * torch.eye(4)
expected[0][1] = 4.0 * torch.eye(4)
expected[1][0] = 6.0 * torch.eye(4)
expected[1][1] = 8.0 * torch.eye(4)

assert hessian.equal(expected), "Error in Hessian values"
```
