import abc

import numpy as np


class Populator:
    def __init__(self, dna: list, dimensions: int) -> None:
        self.dna = dna
        self.dimensions = dimensions

    def populate(self, size, random_state=None):
        rng = np.random.default_rng(random_state)
        return np.array(
            [
                rng.choice(self.dna, size=self.dimensions, replace=False)
                for _ in range(size)
            ]
        )


class Selector(abc.ABC):
    @abc.abstractmethod
    def select(self, pop, /, random_state=None):
        raise NotImplementedError


class Crossover(abc.ABC):
    @abc.abstractmethod
    def crossover(self, pop, /, cxpb=1.0, random_state=None):
        raise NotImplementedError


class Mutator(abc.ABC):
    @abc.abstractmethod
    def mutate(self, pop, /, mutpb=1.0, random_state=None):
        raise NotImplementedError


class Evaluator(abc.ABC):
    @abc.abstractmethod
    def evaluate(self, pop, /):
        raise NotImplementedError


class Terminator(abc.ABC):
    @abc.abstractmethod
    def terminate(self, pop, /) -> bool:
        raise NotImplementedError


class GenerationLimit(Terminator):
    def __init__(self, maxiters: int) -> None:
        self.maxiters = maxiters
        self.current_iter = 0

    def terminate(self, _pop, /) -> bool:
        self.current_iter += 1
        return self.current_iter >= self.maxiters


class EvaluationLimit(Terminator):
    class WithEvaluationCount(Evaluator):
        def __init__(self, evaluator: Evaluator) -> None:
            self._inner = evaluator
            self._nevals = 0

        def nevals(self):
            return self._nevals

        def evaluate(self, pop, /):
            self._nevals += 1
            return self._inner.evaluate(pop)

    def __init__(self, evaluator: Evaluator, maxevals: int) -> None:
        self.maxevals = maxevals
        self._evaluator = self.WithEvaluationCount(evaluator)

    def evaluator(self) -> Evaluator:
        return self._evaluator

    def terminate(self, _pop, /) -> bool:
        return self._evaluator.nevals() >= self.maxevals


class FitnessThresholdLimit(Terminator):
    def __init__(self, evaluator: Evaluator, threshold) -> None:
        self._threshold = threshold
        self._evaluator = evaluator

    def terminate(self, pop, /) -> bool:
        fitnesses = self._evaluator.evaluate(pop)
        return np.min(fitnesses) <= self._threshold


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


class UniformOrderCrossover(Crossover):
    def crossover(self, pop, /, cxpb=1.0, random_state=None):
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


class SwapMutator(Mutator):
    def mutate(self, pop, /, mutpb=1.0, random_state=None):
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


class SimpleGA:
    def __init__(
        self,
        populator: Populator,
        selector: Selector,
        crossover: Crossover,
        mutator: Mutator,
    ) -> None:
        self.populator = populator
        self.selector = selector
        self.crossover = crossover
        self.mutator = mutator

    def run(self, size, cxpb=1.0, mutpb=1.0, random_state=None):
        pop = self.populator.populate(size, random_state=random_state)
        while True:
            pop = self.selector.select(pop, random_state=random_state)
            pop = self.crossover.crossover(pop, cxpb=cxpb, random_state=random_state)
            pop = self.mutator.mutate(pop, mutpb=mutpb, random_state=random_state)
            yield pop

    def optimize(
        self, terminator: Terminator, size, cxpb=1.0, mutpb=1.0, random_state=None
    ):
        steps = self.run(size, cxpb, mutpb, random_state)
        pop = next(steps)
        while not terminator.terminate(pop):
            pop = next(steps)
        return pop


def optimize(
    populator: Populator,
    selector: Selector,
    crossover: Crossover,
    mutator: Mutator,
    terminator: Terminator,
    random_state=None,
):
    algorithm = SimpleGA(
        populator,
        selector,
        crossover,
        mutator,
    )
    return algorithm.optimize(
        terminator, 50, cxpb=0.9, mutpb=0.1, random_state=random_state
    )


class AscendingSequenceEvaluator(Evaluator):
    def evaluate(self, pop):
        fitnesses = []
        for ind in pop:
            fit = len(ind)
            for i in range(1, np.shape(pop)[1]):
                if ind[i - 1] < ind[i]:
                    fit -= 1.0
            fitnesses.append(fit)
        return np.array(fitnesses)


def main():
    dna = [0, 1, 2, 3, 4, 5, 6, 7]
    evaluator = AscendingSequenceEvaluator()
    population = optimize(
        populator=Populator(dna, dimensions=len(dna)),
        selector=TournamentSelection(evaluator, k=3),
        crossover=UniformOrderCrossover(),
        mutator=SwapMutator(),
        terminator=GenerationLimit(100),
        random_state=123,
    )
    fitnesses = evaluator.evaluate(population)
    best = population[np.argmin(fitnesses)]
    bestfit = fitnesses[np.argmin(fitnesses)]
    print(best)
    print(bestfit)


if __name__ == "__main__":
    main()
