import abc
import numpy as np


class Swarm:
    def __init__(self, positions, fitnesses) -> None:
        self.positions = positions
        self.fitnesses = fitnesses

    def best(self):
        return self.positions[np.argmin(self.fitnesses)]

    def bestfit(self):
        return np.min(self.fitnesses)


class ParticleSwarmOptimization(abc.ABC):
    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    @abc.abstractmethod
    def next(self) -> Swarm:
        raise NotImplementedError


class StandardPSO(ParticleSwarmOptimization):
    def __init__(
        self, fitfn, dims, size, c1=1.49618, c2=1.49618, w=0.729844, random_seed=None
    ):
        self.fitfn = fitfn
        self.dims = dims
        self.size = size
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.rng = np.random.default_rng(random_seed)
        self._started = False

    def next(self) -> Swarm:
        if not self._started:
            self._started = True
            self.positions = self.rng.random((self.size, self.dims))
            self.positions_fit = np.fromiter(
                map(self.fitfn, self.positions), dtype=float
            )
            self.velocities = np.zeros_like(self.positions)
            self.pbests = np.copy(self.positions)
            self.pbests_fit = np.copy(self.positions_fit)
            self.gbest = self.positions[np.argmin(self.positions_fit)]
            self.gbest_fit = np.min(self.positions_fit)
            return Swarm(self.positions, self.positions_fit)

        for i, pos in enumerate(self.positions):
            if self.positions_fit[i] < self.pbests_fit[i]:
                self.pbests[i] = pos
                self.pbests_fit[i] = self.positions_fit[i]
            if self.pbests_fit[i] < self.gbest_fit:
                self.gbest = self.pbests[i]
                self.gbest_fit = self.pbests_fit[i]
        r1 = self.rng.random(self.positions.shape)
        r2 = self.rng.random(self.positions.shape)
        self.velocities @= np.diag([self.w] * self.dims)
        self.velocities += (
            self.c1 * r1 * (self.pbests - self.positions)
        )  # cognitive term
        self.velocities += self.c2 * r2 * (self.gbest - self.positions)  # social term
        self.positions += self.velocities
        self.positions_fit = np.fromiter(
            map(self.fitfn, self.positions), dtype=self.positions_fit.dtype
        )
        return Swarm(self.positions, self.positions_fit)


class CooperativePSOSplit(ParticleSwarmOptimization):
    def __init__(
        self,
        fitfn,
        dims: int,
        size: int,
        split_factor: int | None = None,
        c1: float = 1.49618,
        c2: float = 1.49618,
        w: float = 0.729844,
        random_seed=None,
    ) -> None:
        assert (
            split_factor is None or split_factor <= dims
        ), "split factor must be less than or equal to the number of components (ie. K <= n)"
        split_factor = split_factor or dims
        rng = np.random.default_rng(random_seed)
        self.fitfn = fitfn
        self.component_dim = dims // split_factor
        self.ctx = rng.random(dims, dtype=np.float64)

        self.psos = [
            StandardPSO(
                fitfn=lambda x: self._partial_fitfn(x, j),
                dims=self.component_dim,
                size=size,
                c1=c1,
                c2=c2,
                w=w,
                random_seed=rng,
            )
            for j in range(split_factor)
        ]

    def _partial_fitfn(self, x, j):
        solution = np.copy(self.ctx)
        solution[j : j + self.component_dim] = x
        return self.fitfn(solution)

    def next(self) -> Swarm:
        for j, pso in enumerate(self.psos):
            swarm = pso.next()
            self.ctx[j : j + self.component_dim] = swarm.best()
        return Swarm(np.array([self.ctx]), np.array([self.fitfn(self.ctx)]))
