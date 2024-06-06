import abc

import numpy as np


class Crossover(abc.ABC):
    @abc.abstractmethod
    def crossover(self, pop, /, cxpb: float, random_state=None):
        raise NotImplementedError


class UniformOrderCrossover(Crossover):
    def crossover(self, pop, /, cxpb: float, random_state=None):
        rng = np.random.default_rng(random_state)
        probmask = rng.choice([True, False], size=len(pop), p=(cxpb, 1.0 - cxpb))
        pool = pop[probmask]
        p1 = pool[::2]
        p2 = pool[1::2]
        c1 = np.copy(p1)
        c2 = np.copy(p2)
        mask = rng.choice([True, False], size=np.shape(p2))
        for i, m in enumerate(mask):
            c1[i, ~m] = p2[i, np.isin(p2[i], p1[i, m], invert=True)][: len(c1[i, ~m])]
            c2[i, ~m] = p1[i, np.isin(p1[i], p2[i, m], invert=True)][: len(c2[i, ~m])]
        return np.concatenate((pop[~probmask], c1, c2), axis=0)


class Duplicate(Crossover):
    def crossover(self, pop, /, cxpb: float, random_state=None):
        return pop
