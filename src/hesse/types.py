"""Type aliases."""

from typing import Any

import torch

Hessian = Any
Loss = torch.nn.modules.loss._Loss  # pylint: disable=protected-access
