import abc

import numpy as np


class Mutator(abc.ABC):
    @abc.abstractmethod
    def mutate(self, pop, /, mutpb: float, random_state=None):
        raise NotImplementedError


class SwapMutator(Mutator):
    def mutate(self, pop, /, mutpb: float, random_state=None):
        rng = np.random.default_rng(random_state)
        _, m = np.shape(pop)
        newpop = []
        for ind in pop:
            ind = np.copy(ind)
            if rng.random() < mutpb:
                fst, snd = rng.choice(m, size=2, replace=False)
                ind[[fst, snd]] = ind[[snd, fst]]
            newpop.append(ind)
        return np.array(newpop)


class Preserve(Mutator):
    def mutate(self, pop, /, mutpb: float, random_state=None):
        return pop
