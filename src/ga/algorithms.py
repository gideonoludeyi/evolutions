from .populator import Populator
from .selectors import Selector
from .crossovers import Crossover
from .mutators import Mutator
from .terminators import Terminator


class SimpleGA:
    def __init__(
        self,
        populator: Populator,
        selector: Selector,
        crossover: Crossover,
        mutator: Mutator,
        terminator: Terminator,
    ) -> None:
        self.populator = populator
        self.selector = selector
        self.crossover = crossover
        self.mutator = mutator
        self.terminator = terminator

    def run(self, size, cxpb=1.0, mutpb=1.0, random_state=None):
        pop = self.populator.populate(size, random_state=random_state)
        while not self.terminator.terminate(pop):
            pop = self.selector.select(pop, random_state=random_state)
            pop = self.crossover.crossover(pop, cxpb=cxpb, random_state=random_state)
            pop = self.mutator.mutate(pop, mutpb=mutpb, random_state=random_state)
            yield pop

    def optimize(self, size, cxpb=1.0, mutpb=1.0, random_state=None):
        steps = self.run(size, cxpb, mutpb, random_state)
        pop = next(steps)
        try:
            while True:
                pop = next(steps)
        except StopIteration:
            return pop


def optimize(
    populator: Populator,
    selector: Selector,
    crossover: Crossover,
    mutator: Mutator,
    terminator: Terminator,
    size=50,
    cxpb=0.9,
    mutpb=0.1,
    random_state=None,
):
    algorithm = SimpleGA(populator, selector, crossover, mutator, terminator)
    return algorithm.optimize(size, cxpb=cxpb, mutpb=mutpb, random_state=random_state)
