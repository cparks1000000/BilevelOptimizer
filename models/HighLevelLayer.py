import torch.nn as nn
import torch
from torch import Tensor
from torch.nn import functional as f
from torch.nn import init
import math


class TestLinearMod(nn.Module):
    def __init__(self, in_features: int, out_features: int, use_bias: bool = False) -> None:
        super(TestLinearMod, self).__init__()
        # Save nums for forward() method
        self.input_features: int = in_features
        self.output_features: int = out_features

        # Declare weight tensor as a Parameter object
        # Parameter just wraps around tensors to handle some tasks required
        self.weight: nn.Parameter = nn.Parameter(torch.empty(out_features, in_features))
        # Declare bias tensor if required
        if use_bias:
            self.bias: nn.Parameter = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters. This is exactly how they're initialized in std Linear
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return f.linear(input, self.weight, self.bias)
