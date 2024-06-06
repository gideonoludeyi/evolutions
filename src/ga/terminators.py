import abc


class Terminator(abc.ABC):
    @abc.abstractmethod
    def terminate(self, pop, /) -> bool:
        raise NotImplementedError


class StopImmediately(Terminator):
    def terminate(self, _pop, /) -> bool:
        return True


class GenerationLimit(Terminator):
    def __init__(self, maxiters: int) -> None:
        self.maxiters = maxiters
        self.current_iter = 0

    def terminate(self, _pop, /) -> bool:
        self.current_iter += 1
        return self.current_iter >= self.maxiters
