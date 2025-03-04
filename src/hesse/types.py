"""Type aliases."""

import torch

Criterion = torch.nn.modules.loss._Loss  # pylint: disable=protected-access
