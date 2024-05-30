import numpy as np
import matplotlib.pyplot as plt


def pso(fitfn, dims, size, c1=1.49618, c2=1.49618, w=0.729844, random_seed=None):
    rng = np.random.default_rng(random_seed)
    positions = rng.random((size, dims))
    positions_fit = np.fromiter(map(fitfn, positions), dtype=float)
    velocities = np.zeros_like(positions)
    pbests = np.copy(positions)
    pbests_fit = np.copy(positions_fit)
    gbest = positions[np.argmin(positions_fit)]
    gbest_fit = np.min(positions_fit)
    yield positions
    while True:
        for i, pos in enumerate(positions):
            if positions_fit[i] < pbests_fit[i]:
                pbests[i] = pos
                pbests_fit[i] = positions_fit[i]
            if pbests_fit[i] < gbest_fit:
                gbest = pbests[i]
                gbest_fit = pbests_fit[i]
        r1 = rng.random(positions.shape)
        r2 = rng.random(positions.shape)
        velocities @= np.diag([w] * dims)
        velocities += c1 * r1 * (pbests - positions)  # cognitive term
        velocities += c2 * r2 * (gbest - positions)  # social term
        positions += velocities
        positions_fit = np.fromiter(map(fitfn, positions), dtype=positions_fit.dtype)
        yield positions


def cpso_s(fitfn, dims, size, c1=1.49618, c2=1.49618, w=0.729844, random_seed=None):
    ctx = np.zeros(dims, dtype=np.float64)

    def unifit(ctx, x, j):
        solution = np.copy(ctx)
        solution[j] = x[0]
        return fitfn(solution)

    generators = [
        pso(
            fitfn=lambda x: unifit(ctx, x, j),
            dims=1,
            size=size,
            c1=c1,
            c2=c2,
            w=w,
            random_seed=random_seed,
        )
        for j in range(dims)
    ]
    while True:
        for j, generator in enumerate(generators):
            positions = next(generator)
            best = positions[
                np.argmin(
                    np.fromiter(
                        map(lambda x: unifit(ctx, x, j), positions), dtype=float
                    )
                )
            ]
            ctx[j] = best[0]
        yield np.array([ctx])


def rosenbrock(x):
    x = np.asarray(x)
    n = x.shape[0]
    res = 0.0
    for i in range(1, n // 2):
        res += 100 * (x[2 * i] - x[2 * i - 1] ** 2) ** 2 + (1 - x[2 * i - 1]) ** 2
    return res


def quadric(x):
    x = np.asarray(x)
    n = x.shape[0]
    res = 0.0
    for i in range(1, n + 1):
        res += np.sum(x[:i]) ** 2
    return res


def ackley(x):
    x = np.asarray(x)
    n = x.shape[0]
    return (
        -20 * np.exp(-0.2 * np.sqrt((1 / n) * np.sum(x**2)))
        - np.exp((1 / n) * np.sum(np.cos(2 * np.pi * x)))
        + 20
        + np.e
    )


def rastrigin(x):
    x = np.asarray(x)
    return np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10)


def griewank(x):
    x = np.asarray(x)
    n = x.shape[0]
    return (
        (1 / 4000) * np.sum(x**2)
        - np.prod(np.cos(x / np.sqrt(np.arange(1, n + 1))))
        + 1
    )


def plot(fitfn, generator, fname="plot.png"):
    fits = [
        np.min(np.fromiter(map(fitfn, positions), dtype=float))
        for positions, _ in zip(generator, range(50))
    ]
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(fits)), fits)
    fig.savefig(fname)
    plt.close(fig)


fitness_functions = dict(
    rosenbrock=rosenbrock,
    quadric=quadric,
    ackley=ackley,
    rastrigin=rastrigin,
    griewank=griewank,
)


def main():
    for name, fitfn in fitness_functions.items():
        print(f"Run: {name}")
        generator = cpso_s(fitfn, dims=30, size=20, random_seed=123)
        plot(fitfn, generator, fname=f"tmp/{name}.png")
