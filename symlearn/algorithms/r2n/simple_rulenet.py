import torch
from torch import nn

import symlearn.algorithms.r2n.mask as wm
from symlearn.algorithms.r2n.base import RuleNetwork


class SimpleRuleNet(RuleNetwork):
    """
    Class that assembles the rule network part of r2n. It initializes and configures the conjunction and disjunction layer. 
    """
    def __init__(self, n_conditions, n_rules, cooling=True):
        super(SimpleRuleNet, self).__init__()
        self.n_rules = n_rules
        self.n_conditions = n_conditions
        self.temp = 1
        self.cooling = cooling
        self.weightmask_1 = wm.SimpleWeightMask(self.n_rules, self.n_conditions, self.temp, self.cooling)
        self.weightmask_2 = wm.SimpleWeightMask(1, self.n_rules, self.temp, self.cooling)
        self.network = nn.Sequential(
            ConjunctionLayer(self.weightmask_1),
            DisjunctionLayer(self.weightmask_2),
        )

    def forward(self, x):
        return self.network(x)

    def extract_ruleset(self):
        ruleset = []
        w_conj = torch.heaviside(self.network[0].mask() - torch.tensor(0.5),
                                 torch.ones_like(self.network[0].mask()))
        w_disj = torch.heaviside(self.network[1].mask() - torch.tensor(0.5),
                                 torch.ones_like(self.network[1].mask()))
        for d_idx, d_weight in enumerate(torch.flatten(w_disj).tolist()):
            if d_weight > 0:
                rule = w_conj.tolist()[d_idx]
                if sum(rule) > 0:
                    ruleset.append(rule)
        return ruleset

    def get_penalty(self):
        and_masks = self.network[0].mask()
        or_masks = self.network[1].mask()

        return (or_masks.sum() + and_masks.sum()) / self.n_rules

    def update_temp(self, temp):
        self.weightmask_1.set_temp(temp)
        self.weightmask_2.set_temp(temp)


class ConjunctionLayer(nn.Module):
    """
    Conjuction layer (And layer)
    """
    def __init__(self, mask):
        super(ConjunctionLayer, self).__init__()
        self.mask = mask

    def forward(self, x):
        neg_x = torch.ones_like(x) - x
        prod = torch.matmul(neg_x, torch.transpose(self.mask().float(), 0, 1))
        return torch.ones_like(prod) - torch.clamp(prod, max=1)


class DisjunctionLayer(nn.Module):
    """
    Disjunction layer (OR layer)
    """
    def __init__(self, mask):
        super(DisjunctionLayer, self).__init__()
        self.mask = mask

    def forward(self, x):
        return torch.clamp(torch.matmul(self.mask(), torch.transpose(x, 0, 1)), max=1)
