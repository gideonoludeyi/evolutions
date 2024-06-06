import pathlib
import argparse
import io

import ga
import numpy as np
from scipy.spatial.distance import pdist, squareform


from .specification import (
    CrossoverFactory,
    MutatorFactory,
    SelectorFactory,
    Specification,
    TerminatorFactory,
)


def parse_coord(raw: str) -> tuple[int, float, float]:
    iden, x, y = raw.split(" ")
    return int(iden), float(x), float(y)


def parse_tsp_coords(inp: io.TextIOWrapper):
    lines = list(map(str.strip, inp.readlines()))
    startidx = lines.index("NODE_COORD_SECTION") + 1
    endidx = lines.index("EOF", startidx)
    datalines = lines[startidx:endidx]
    return [parse_coord(line) for line in datalines]


class TravelingSalesmanProblem(ga.Evaluator):
    def __init__(self, coords) -> None:
        self.coords = np.asarray([coord[1:3] for coord in coords])

    def evaluate(self, pop, /):
        lengths = []
        for indices in pop:
            indices = np.subtract(indices, 1)  # convert from 1- to 0-based indexing
            dist_matrix = squareform(pdist(self.coords[indices]))
            pathlen = 0.0
            for i in range(len(indices) - 1):
                src = indices[i]
                dst = indices[i + 1]
                pathlen += dist_matrix[src, dst]
            lengths.append(pathlen)
        return np.array(lengths)


parser = argparse.ArgumentParser("tsp")
parser.add_argument("inputfile", type=argparse.FileType("r"))
parser.add_argument(
    "--specfile",
    dest="specfile",
    type=pathlib.Path,
    required=False,
    default=pathlib.Path("spec.toml"),
)
parser.add_argument(
    "-p", "--popsize", dest="popsize", type=int, required=False, default=50
)
parser.add_argument(
    "-c", "--crossover-rate", dest="cxpb", type=float, required=False, default=0.9
)
parser.add_argument(
    "-m", "--mutation-rate", dest="mutpb", type=float, required=False, default=0.1
)
parser.add_argument(
    "--seed", dest="random_seed", type=int, required=False, default=None
)
parser.add_argument(
    "--selector", dest="selector_id", type=str, required=False, default=None
)
parser.add_argument(
    "--crossover", dest="crossover_id", type=str, required=False, default=None
)
parser.add_argument(
    "--mutator", dest="mutator_id", type=str, required=False, default=None
)
parser.add_argument(
    "--terminator", dest="terminator_id", type=str, required=False, default=None
)


def selector(factory: SelectorFactory, id_: str | None, evaluator: ga.Evaluator):
    if id_ is None:
        return factory.default(evaluator)
    return factory.get(id_, evaluator)


def crossover(factory: CrossoverFactory, id_: str | None):
    if id_ is None:
        return factory.default()
    return factory.get(id_)


def mutator(factory: MutatorFactory, id_: str | None):
    if id_ is None:
        return factory.default()
    return factory.get(id_)


def terminator(factory: TerminatorFactory, id_: str | None):
    if id_ is None:
        return factory.default()
    return factory.get(id_)


def main():
    args = parser.parse_args()
    with args.inputfile as f:
        coords = parse_tsp_coords(f)
    evaluator = TravelingSalesmanProblem(coords)
    spec_factory_factory = Specification.parse_toml(args.specfile)
    initializer = ga.Populator([coord[0] for coord in coords], dimensions=len(coords))
    population = ga.optimize(
        size=args.popsize,
        cxpb=args.cxpb,
        mutpb=args.mutpb,
        populator=initializer,
        selector=selector(spec_factory_factory.selector(), args.selector_id, evaluator),
        crossover=crossover(spec_factory_factory.crossover(), args.crossover_id),
        mutator=mutator(spec_factory_factory.mutator(), args.mutator_id),
        terminator=terminator(spec_factory_factory.terminator(), args.terminator_id),
        random_state=args.random_seed,
    )
    fitnesses = evaluator.evaluate(population)
    print("Best Ind", population[np.argmin(fitnesses)])
    print("Best Fit", np.min(fitnesses))
    print("Mean Fit", np.mean(fitnesses))
    print("Std. Fit", np.std(fitnesses))


if __name__ == "__main__":
    main()
