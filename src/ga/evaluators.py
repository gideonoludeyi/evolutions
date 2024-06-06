import abc


class Evaluator(abc.ABC):
    @abc.abstractmethod
    def evaluate(self, pop, /):
        raise NotImplementedError
