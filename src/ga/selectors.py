import abc
import numpy as np

from .evaluators import Evaluator


class Selector(abc.ABC):
    @abc.abstractmethod
    def select(self, pop, /, random_state=None):
        raise NotImplementedError


class TournamentSelection(Selector):
    def __init__(self, evaluator: Evaluator, k: int = 3, with_replacement=True) -> None:
        self.evaluator = evaluator
        self.k = k
        self.with_replacement = with_replacement

    def select(self, pop, /, random_state=None):
        rng = np.random.default_rng(random_state)
        fitnesses = self.evaluator.evaluate(pop)
        newpop = []
        for _ in range(len(pop)):
            indices = rng.choice(
                len(pop), size=self.k, replace=self.with_replacement, shuffle=True
            )
            bestidx = min(indices, key=lambda idx: fitnesses[idx])
            bestind = pop[bestidx]
            newpop.append(bestind)
        return np.array(newpop)


class SelectAll(Selector):
    def select(self, pop, /, random_state=None):
        return pop
