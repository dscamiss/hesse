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

* *Model Hessians* (these are the Hessians of a given model with respect to its trainable parameters);
* *Loss function Hessians* (these are the Hessians of `loss(model(inputs), target)` with respect to `model`'s trainable parameters).

This is achieved with user-friendly wrappers for `torch.func` transforms.

# Installation

```bash
git clone https://github.com/dscamiss/hesse/
pip install hesse
```

# Example

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
            x: First input tensor of shape (m, n).
            y: Second input tensor of shape (m, n).

        Returns:
            The matrix

            [ tr(A^t A) x_{    0, :}   tr(B^t B) y_{    0, :} ]
            [ tr(A^t A) x_{    1, :}   tr(B^t B) y_{    1, :} ]
            [           :                        :            ]
            [ tr(A^t A) x_{m - 1, :}   tr(B^t B) y_{m - 1, :} ].
        """
        rows_1 = torch.trace(self.A.T @ self.A) * x
        rows_2 = torch.trace(self.B.T @ self.B) * y
        return torch.hstack((rows_1, rows_2))
```

Make an instance of `MimoModel` along with some inputs.

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

Computing the Hessian matrix of `model` is a one-liner:

```python
hessian = hesse.model_hessian_matrix(model=model, inputs=(x, y))
```

We can verify the correctness of the result.

```python
model_output = model(x, y)
expected = torch.zeros(model_output.shape + torch.Size([8, 8]))
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

# TODO

- [X] Add remaining test cases
- [ ] Add diagonal-only sharpness computation 
- [ ] Sparse storage where appropriate
- [X] Test with multi-output model
- [X] Test batch functions with batch data
- [ ] Remove redundancy in test code
- [X] Add a few examples
