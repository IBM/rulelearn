from abc import ABC, abstractmethod

from torch import nn


class RuleNetwork(nn.Module, ABC):
    """
    Encapsulates a neural network that corresponds syntactically to some ruleset.
    """

    @abstractmethod
    def extract_ruleset(self):
        """ Extract a symbolic representation of the learned ruleset. The semantics of the ruleset must be provided
        externally.

        :return: A 2D list representing a ruleset in Disjunctive Normal Form. An inner list takes values from {-1, 0, 1}
         representing a single rule, indicating the inclusion of the negated condition, exclusion,
         and inclusion of the condition, respectively.
         E.g. [[1, 0, 0, -1], [0, 1, 0, 0]] represents the ruleset
         "(condition0 ^ not condition3) v condition1" for a rule network with 4 input conditions.
        """
        raise NotImplementedError


class PredicateLearningLayer(nn.Module, ABC):
    @abstractmethod
    def get_predicates(self, *args, **kwargs):
        raise NotImplementedError
