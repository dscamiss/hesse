"""Type aliases."""

import torch
from jaxtyping import Num
from torch import Tensor

HessianDict = dict[str, dict[str, Num[Tensor, "..."]]]
Criterion = torch.nn.modules.loss._Loss  # pylint: disable=protected-access
ParamDict = dict[str, Tensor]
