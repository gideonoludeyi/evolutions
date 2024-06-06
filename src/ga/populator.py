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
