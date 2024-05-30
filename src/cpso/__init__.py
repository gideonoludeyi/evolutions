import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

from . import algorithms, objectives


def pso(algorithm: algorithms.ParticleSwarmOptimization, niters=50):
    assert niters > 0, "niters must be positive"
    best_swarm = algorithm.next()
    for _ in range(niters):
        best_swarm = min(
            best_swarm, algorithm.next(), key=lambda swarm: swarm.bestfit()
        )
    return best_swarm.best()


def plot(
    algorithm: algorithms.ParticleSwarmOptimization, niters=50, out="tmp/plot.png"
):
    fits = [swarm.bestfit() for swarm, _ in zip(algorithm, range(niters))]
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(fits)), fits)
    ax.set_title("Fitness over time")
    ax.set_ylabel("Fitness")
    ax.set_xlabel("Iterations")
    fig.savefig(out)
    plt.close(fig)


root_parser = argparse.ArgumentParser("pso")
subparser_creator = root_parser.add_subparsers(required=True)


def pso_cli_executor(args: argparse.Namespace):
    algorithm = algorithms.StandardPSO(
        objectives.functions[args.fitness],
        dims=args.ndims,
        size=args.popsize,
        c1=args.c1,
        c2=args.c2,
        w=args.inertia_weight,
        random_seed=args.random_seed,
    )
    solution = pso(algorithm, niters=args.niter)
    print(solution, file=args.outputfile)


pso_parser = subparser_creator.add_parser("run")
pso_parser.add_argument(
    "-n", "--ndims", dest="ndims", type=int, required=False, default=30
)
pso_parser.add_argument(
    "-p", "--popsize", dest="popsize", type=int, required=False, default=20
)
pso_parser.add_argument(
    "-g", "--generations", dest="niter", type=int, required=False, default=50
)
pso_parser.add_argument(
    "-c", "--cognitive", dest="c1", type=float, required=False, default=1.49618
)
pso_parser.add_argument(
    "-s", "--social", dest="c2", type=float, required=False, default=1.49618
)
pso_parser.add_argument(
    "-w",
    "--inertia",
    dest="inertia_weight",
    type=float,
    required=False,
    default=0.729844,
)
pso_parser.add_argument(
    "--fitness",
    dest="fitness",
    type=str,
    choices=objectives.functions.keys(),
    required=False,
    default="rosenbrock",
)
pso_parser.add_argument(
    "--seed",
    dest="random_seed",
    type=int,
    required=False,
    default=None,
    help="value with which to seed the random number generator for reproducible results",
)
pso_parser.add_argument(
    "-o",
    "--output",
    dest="outputfile",
    type=argparse.FileType("w"),
    required=False,
    default=sys.stdout,
    help="the file which the solution should be written to [default: stdout]",
)
pso_parser.set_defaults(__exec__=pso_cli_executor)


def plot_pso_cli_executor(args: argparse.Namespace):
    algorithm = algorithms.StandardPSO(
        objectives.functions[args.fitness],
        dims=args.ndims,
        size=args.popsize,
        c1=args.c1,
        c2=args.c2,
        w=args.inertia_weight,
        random_seed=args.random_seed,
    )
    plot(algorithm, niters=args.niter, out=args.outputfile)


plot_parser = subparser_creator.add_parser("plot")
plot_parser.add_argument(
    "-n", "--ndims", dest="ndims", type=int, required=False, default=30
)
plot_parser.add_argument(
    "-p", "--popsize", dest="popsize", type=int, required=False, default=20
)
plot_parser.add_argument(
    "-g", "--generations", dest="niter", type=int, required=False, default=50
)
plot_parser.add_argument(
    "-c", "--cognitive", dest="c1", type=float, required=False, default=1.49618
)
plot_parser.add_argument(
    "-s", "--social", dest="c2", type=float, required=False, default=1.49618
)
plot_parser.add_argument(
    "-w",
    "--inertia",
    dest="inertia_weight",
    type=float,
    required=False,
    default=0.729844,
)
plot_parser.add_argument(
    "--fitness",
    dest="fitness",
    type=str,
    choices=objectives.functions.keys(),
    required=False,
    default="rosenbrock",
)
plot_parser.add_argument(
    "--seed",
    dest="random_seed",
    type=int,
    required=False,
    default=None,
    help="value with which to seed the random number generator for reproducible results",
)
plot_parser.add_argument(
    "-o",
    "--output",
    dest="outputfile",
    type=argparse.FileType("wb"),
    required=False,
    default=sys.stdout.buffer,
    help="the file which the plot should be written to [default: stdout]",
)
plot_parser.set_defaults(__exec__=plot_pso_cli_executor)


def main():
    args = root_parser.parse_args()
    args.__exec__(args)


if __name__ == "__main__":
    main()
