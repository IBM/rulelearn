from abc import ABC, abstractmethod

import torch
from torch import nn

from symlearn.algorithms.r2n.functions import DifferentiableHeaviside


class WeightMask(nn.Module, ABC):
    """
    Abstract class for the mask that is applied to the weight matrix of the NN. 
    """

    def __init__(self, dim_out, dim_in):
        super(WeightMask, self).__init__()
        self.dim_out = dim_out
        self.dim_in = dim_in
        self.loc = torch.nn.Parameter(torch.randn(self.dim_out, self.dim_in), requires_grad=True)

    @abstractmethod
    def forward(self):
        pass


class SimpleWeightMask(WeightMask):
    """
    Binary weight mask that is used in the AND/OR layer. Note that depening of the Boolean value cooling, we either implement a cooled sigmoid or stricly binary differentiable heavyside function. 
    """

    def __init__(self, dim_out, dim_in, temp, cooling):
        super(SimpleWeightMask, self).__init__(dim_out, dim_in)
        self.cooling = cooling
        self.temp = temp

    def forward(self):
        if self.cooling is True:
            return torch.sigmoid((1 / self.temp) * self.loc)
        else:
            return DifferentiableHeaviside.apply(self.loc)

    def set_temp(self, temp):
        self.temp = temp

    def get_temp(self):
        return self.temp


class DeterministicWeightMask(WeightMask):
    """
    Simple deterministic weight mask. 
    """

    def __init__(self, mask):
        super(DeterministicWeightMask, self).__init__(mask.size()[0], mask.size()[1])
        self.mask = mask

    def forward(self):
        return self.mask
