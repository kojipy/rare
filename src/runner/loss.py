from typing import Optional

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


class CustomCrossEntropy(CrossEntropyLoss):
    def __init__(
        self,
        weight: Optional[Tensor] = None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = "mean",
        label_smoothing: float = 0,
    ) -> None:
        super().__init__(
            weight, size_average, ignore_index, reduce, reduction, label_smoothing
        )

    def forward(self, batched_input: Tensor, batched_target: Tensor) -> Tensor:
        # inputs = torch.zeros_like(
        #     batched_input.view(-1, batched_input.shape[-1]), requires_grad=True
        # )
        # targets = torch.zeros_like(batched_target.view(-1))

        inputs = []
        targets = []

        # curr_idx = 0
        for batch_i, target in enumerate(batched_target):
            use4loss = int(target.nonzero().max()) + 1
            # targets[curr_idx : curr_idx + use4loss] = target[:use4loss]
            # inputs[curr_idx : curr_idx + use4loss, :] = batched_input[
            #     batch_i, :use4loss
            # ]

            # curr_idx += use4loss
            targets.append(target[:use4loss])
            inputs.append(batched_input[batch_i, :use4loss])

        # use4loss = int(targets.nonzero().max()) + 1
        # targets = targets[:use4loss]
        # inputs = inputs[:use4loss, :]

        targets = torch.cat(targets, dim=0)
        inputs = torch.cat(inputs, dim=0)

        return F.cross_entropy(
            inputs,
            targets,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )
