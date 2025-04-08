"""Type aliases."""

from typing import Iterable, Optional, Union

import torch
from jaxtyping import Num
from torch import Tensor

Criterion = torch.nn.modules.loss._Loss  # pylint: disable=protected-access

HessianDict = dict[str, dict[str, Num[Tensor, "..."]]]
Inputs = Union[Num[Tensor, "..."], tuple[Num[Tensor, "..."], ...]]
Target = Num[Tensor, "..."]

BatchHessianDict = dict[str, dict[str, Num[Tensor, "b ..."]]]
BatchInputs = Union[Num[Tensor, "b ..."], tuple[Num[Tensor, "b ..."], ...]]
BatchTarget = Num[Tensor, "b ..."]

HessianMatrix = Num[Tensor, "n n"]
BatchHessianMatrix = Num[Tensor, "b n n"]

Params = Optional[Iterable[str]]
