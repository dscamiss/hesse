"""Type aliases."""

import torch
from jaxtyping import Num
from torch import Tensor

BatchHessianDict = dict[str, dict[str, Num[Tensor, "b ..."]]]
HessianDict = dict[str, dict[str, Num[Tensor, "..."]]]
Loss = torch.nn.modules.loss._Loss  # pylint: disable=protected-access
