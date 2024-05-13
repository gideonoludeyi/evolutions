import sys
import io
import argparse
import numpy as np
import matplotlib.pyplot as plt


def fit(solution, min_=-np.inf, max_=np.inf):
    x, y = solution
    return np.where(
        (x < min_) | (x > max_) | (y < min_) | (y > max_),
        np.inf,  # use `-np.inf` for a maximization problem
        (x - 3.14) ** 2 + (y - 2.72) ** 2 + np.sin(3 * x + 1.41) + np.sin(4 * y - 1.73),
    )


def pso(
    popsize=10,
    maxiter=50,
    c1=1.0,
    c2=0.5,
    w=None,
    min_=0.0,
    max_=5.0,
    fitfn=fit,
    random_state=None,
):
    w = w or (0.1, 0.1)
    rng = np.random.default_rng(random_state)
    d = len(w)
    W = np.diag(w)
    positions = rng.uniform(min_, max_, (popsize, d))
    velocities = np.zeros(np.shape(positions))
    pbest = np.copy(positions)
    gbest = np.array([min(pbest, key=fitfn)])
    for _ in range(maxiter):
        # update personal best positions
        cond = np.less_equal(
            np.fromiter(map(fitfn, positions), dtype=np.float32),
            np.fromiter(map(fitfn, pbest), dtype=np.float32),
        )
        mask = np.tile(cond.reshape(-1, 1), (1, d))
        pbest = np.where(mask, positions, pbest)
        # update neighborhood best position
        gbest = np.array([min(pbest, key=fitfn)])
        # update velocities
        u1 = rng.random(np.shape(positions))
        u2 = rng.random(np.shape(positions))
        velocities @= W  # in-place matrix multiplication
        velocities += c1 * u1 * (pbest - positions)  # cognitive term
        velocities += c2 * u2 * (gbest - positions)  # social term
        # update positions
        positions += velocities
    return gbest


def pso_cli_executor(args: argparse.Namespace):
    solution = pso(
        c1=args.c1,
        c2=args.c2,
        maxiter=args.maxiter,
        w=args.inertia_weight,
        popsize=args.popsize,
        random_state=args.random_seed,
    )
    print(solution)


def plot_pso(c1, c2, maxiters, w=None, popsize=10, random_state=None):
    fig, axes = plt.subplots(*np.shape(maxiters), figsize=(12, 10))
    for ax, maxiter in zip(np.array(axes).flatten(), np.array(maxiters).flatten()):
        solutions = pso(
            c1=c1,
            c2=c2,
            w=w,
            popsize=popsize,
            maxiter=maxiter,
            fitfn=lambda s: fit(s, min_=0, max_=5),
            random_state=random_state,
        )  # random_state=4 leads to local minima
        x, y = np.array(np.meshgrid(np.linspace(0, 5, 100), np.linspace(0, 5, 100)))
        z = fit([x, y], min_=0, max_=5)
        x_min = x.ravel()[z.argmin()]
        y_min = y.ravel()[z.argmin()]
        plt.figure(figsize=(8, 6))
        img = ax.imshow(
            z, extent=[0, 5, 0, 5], origin="lower", cmap="viridis", alpha=0.5
        )
        fig.colorbar(img)
        ax.set_title(f"g={maxiter}")
        ax.plot([x_min], [y_min], marker="x", markersize=4, color="white")
        for s in solutions:
            ax.plot(s[0], s[1], marker=".", markersize=4, color="red")
        contours = ax.contour(x, y, z, 10, colors="black", alpha=0.4)
        ax.clabel(contours, inline=True, fontsize=8, fmt="%.0f")
    return fig


def plot_pso_cli_executor(args: argparse.Namespace):
    fig = plot_pso(
        c1=args.c1,
        c2=args.c2,
        maxiters=[[args.maxiter]],
        w=args.inertia_weight,
        popsize=args.popsize,
        random_state=args.random_seed,
    )
    fig.savefig(fname=args.outputfile)


root_parser = argparse.ArgumentParser("pso2d")
subparser_creator = root_parser.add_subparsers(required=True)

pso_parser = subparser_creator.add_parser("run")
pso_parser.add_argument(
    "-p", "--popsize", dest="popsize", type=int, required=False, default=10
)
pso_parser.add_argument(
    "-g", "--iterations", dest="maxiter", type=int, required=False, default=50
)
pso_parser.add_argument(
    "-c", "--cognitive", dest="c1", type=float, required=False, default=1.0
)
pso_parser.add_argument(
    "-s", "--social", dest="c2", type=float, required=False, default=0.5
)
pso_parser.add_argument(
    "-w",
    "--inertia",
    action="append",
    dest="inertia_weight",
    type=float,
    required=False,
    default=[0.1, 0.1],
)
pso_parser.add_argument(
    "--seed",
    dest="random_seed",
    type=int,
    required=False,
    default=None,
    help="value with which to seed the random number generator for reproducible results",
)
pso_parser.set_defaults(__exec__=pso_cli_executor)


plot_parser = subparser_creator.add_parser("plot")
plot_parser.add_argument(
    "-p", "--popsize", dest="popsize", type=int, required=False, default=10
)
plot_parser.add_argument(
    "-g", "--iterations", dest="maxiter", type=int, required=False, default=50
)
plot_parser.add_argument(
    "-c", "--cognitive", dest="c1", type=float, required=False, default=1.0
)
plot_parser.add_argument(
    "-s", "--social", dest="c2", type=float, required=False, default=0.5
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
    type=str,
    required=False,
    default="plot.png",
    help="the filepath which the plot should be written",
)
plot_parser.set_defaults(__exec__=plot_pso_cli_executor)


def main():
    args = root_parser.parse_args()
    args.__exec__(args)
