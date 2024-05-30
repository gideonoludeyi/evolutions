import numpy as np


def rosenbrock(x):
    x = np.asarray(x)
    n = x.shape[0]
    return np.sum(
        [
            100 * (x[2 * i] - x[2 * i - 1] ** 2) ** 2 + (1 - x[2 * i - 1]) ** 2
            for i in range(1, n // 2)
        ]
    )


def quadric(x):
    x = np.asarray(x)
    n = x.shape[0]
    return np.sum([np.sum(x[:i]) ** 2 for i in range(1, n + 1)])


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


functions = dict(
    rosenbrock=rosenbrock,
    quadric=quadric,
    ackley=ackley,
    rastrigin=rastrigin,
    griewank=griewank,
)
