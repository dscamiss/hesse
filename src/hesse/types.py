"""Type aliases."""

import torch
from jaxtyping import Num
from torch import Tensor

Hessian = dict[str, dict[str, Num[Tensor, "..."]]]
Loss = torch.nn.modules.loss._Loss  # pylint: disable=protected-access
