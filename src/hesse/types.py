"""Type aliases."""

from typing import Iterable, Optional, Union

import torch
from jaxtyping import Num
from torch import Tensor

Criterion = torch.nn.modules.loss._Loss  # pylint: disable=protected-access

HessianDict = dict[str, dict[str, Num[Tensor, "..."]]]
Inputs = Union[Tensor, tuple[Tensor, ...]]
Target = Union[Tensor, tuple[Tensor, ...]]

BatchHessianDict = dict[str, dict[str, Num[Tensor, "b ..."]]]
BatchInputs = Union[Num[Tensor, "b ..."], tuple[Num[Tensor, "b ..."], ...]]
BatchTarget = Num[Tensor, "b ..."]

Params = Optional[Iterable[str]]
