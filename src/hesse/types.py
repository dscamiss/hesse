"""Type aliases."""

from typing import Iterable, Optional, Union

import torch
from jaxtyping import Num
from torch import Tensor

Criterion = torch.nn.modules.loss._Loss  # pylint: disable=protected-access

HessianDict = dict[str, dict[str, Num[Tensor, "..."]]]
Params = Optional[Iterable[str]]
Inputs = Union[Tensor, tuple[Tensor, ...]]
BatchInputs = Union[Num[Tensor, "b ..."], tuple[Num[Tensor, "b ..."], ...]]
Target = Union[Tensor, tuple[Tensor, ...]]
BatchTarget = Num[Tensor, "b ..."]
