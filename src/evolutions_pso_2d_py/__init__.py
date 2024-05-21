import argparse
import sys
import numpy as np

from . import pso, plot


def fit(solution, min_=-np.inf, max_=np.inf):
    x, y = solution
    return np.where(
        (x < min_) | (x > max_) | (y < min_) | (y > max_),
        np.inf,  # use `-np.inf` for a maximization problem
        (x - 3.14) ** 2 + (y - 2.72) ** 2 + np.sin(3 * x + 1.41) + np.sin(4 * y - 1.73),
    )


root_parser = argparse.ArgumentParser("pso2d")
subparser_creator = root_parser.add_subparsers(required=True)


def pso_cli_executor(args: argparse.Namespace):
    solution = pso.pso_best(
        fit,
        c1=args.c1,
        c2=args.c2,
        maxiter=args.maxiter,
        w=args.inertia_weight,
        popsize=args.popsize,
        random_state=args.random_seed,
    )
    print(f"Best Solution: ({solution[0]}, {solution[1]})", file=args.outputfile)


pso_parser = subparser_creator.add_parser("run")
pso_parser.add_argument(
    "-p", "--popsize", dest="popsize", type=int, required=False, default=20
)
pso_parser.add_argument(
    "-g", "--iterations", dest="maxiter", type=int, required=False, default=50
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
    action="append",
    dest="inertia_weight",
    type=float,
    required=False,
    default=None,
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
    plot.plot_pso(
        fit,
        c1=args.c1,
        c2=args.c2,
        maxiter=args.maxiter,
        w=args.inertia_weight,
        popsize=args.popsize,
        random_state=args.random_seed,
        out=args.outputfile,
    )


plot_parser = subparser_creator.add_parser("plot")
plot_parser.add_argument(
    "-p", "--popsize", dest="popsize", type=int, required=False, default=20
)
plot_parser.add_argument(
    "-g", "--iterations", dest="maxiter", type=int, required=False, default=50
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
    action="append",
    dest="inertia_weight",
    type=float,
    required=False,
    default=None,
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
