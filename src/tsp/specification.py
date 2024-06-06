import os
from typing import Any

import tomllib

from ga.evaluators import Evaluator
from ga.crossovers import Crossover, Duplicate, UniformOrderCrossover
from ga.mutators import Mutator, Preserve, SwapMutator
from ga.selectors import SelectAll, Selector, TournamentSelection
from ga.terminators import GenerationLimit, StopImmediately, Terminator


class SelectorFactory:
    def __init__(self, specs, default: Selector | None = None) -> None:
        self.specs = specs
        self._default = default or SelectAll()

    def default(self, evaluator: Evaluator) -> Selector:
        spec = self.specs.get("default", None)
        if spec is None:
            return self._default
        else:
            return self._from_spec(spec, evaluator)

    def get(self, id_: str, evaluator: Evaluator) -> Selector:
        if id_ not in self.specs:
            raise ValueError("invalid selector id", id_)
        spec = self.specs[id_]
        return self._from_spec(spec, evaluator)

    def _from_spec(self, spec, evaluator: Evaluator) -> Selector:
        type_ = spec["type"]
        if type_ == "tournament":
            return TournamentSelection(evaluator, k=spec["k"])
        else:
            raise ValueError("invalid selector type", type_)


class CrossoverFactory:
    def __init__(self, specs, default: Crossover | None = None) -> None:
        self.specs = specs
        self._default = default or Duplicate()

    def default(self) -> Crossover:
        spec = self.specs.get("default", None)
        if spec is None:
            return self._default
        else:
            return self._from_spec(spec)

    def get(self, id_: str) -> Crossover:
        if id_ not in self.specs:
            raise ValueError("invalid crossover id", id_)
        spec = self.specs[id_]
        return self._from_spec(spec)

    def _from_spec(self, spec) -> Crossover:
        type_ = spec["type"]
        if type_ == "uniform-order-crossover":
            return UniformOrderCrossover()
        else:
            raise ValueError("invalid crossover type", type_)


class MutatorFactory:
    def __init__(self, specs, default: Mutator | None = None) -> None:
        self.specs = specs
        self._default = default or Preserve()

    def default(self) -> Mutator:
        spec = self.specs.get("default", None)
        if spec is None:
            return self._default
        else:
            return self._from_spec(spec)

    def get(self, id_: str) -> Mutator:
        if id_ not in self.specs:
            raise ValueError("invalid mutator id", id_)
        spec = self.specs[id_]
        return self._from_spec(spec)

    def _from_spec(self, spec) -> Mutator:
        type_ = spec["type"]
        if type_ == "swap-mutator":
            return SwapMutator()
        else:
            raise ValueError("invalid mutator type", type_)


class TerminatorFactory:
    def __init__(self, specs, default: Terminator | None = None) -> None:
        self.specs = specs
        self._default = default or StopImmediately()

    def default(self) -> Terminator:
        spec = self.specs.get("default", None)
        if spec is None:
            return self._default
        else:
            return self._from_spec(spec)

    def get(self, id_: str) -> Terminator:
        if id_ not in self.specs:
            raise ValueError("invalid terminator id", id_)
        spec = self.specs[id_]
        return self._from_spec(spec)

    def _from_spec(self, spec) -> Terminator:
        type_ = spec["type"]
        if type_ == "generation-limit":
            return GenerationLimit(spec["n"])
        else:
            raise ValueError("invalid terminator type", type_)


class Specification:
    def __init__(
        self,
        selectors: SelectorFactory,
        crossovers: CrossoverFactory,
        mutators: MutatorFactory,
        terminators: TerminatorFactory,
    ) -> None:
        self.selector_factory = selectors
        self.crossover_factory = crossovers
        self.mutator_factory = mutators
        self.terminator_factory = terminators

    def selector(self) -> SelectorFactory:
        return self.selector_factory

    def crossover(self) -> CrossoverFactory:
        return self.crossover_factory

    def mutator(self) -> MutatorFactory:
        return self.mutator_factory

    def terminator(self) -> TerminatorFactory:
        return self.terminator_factory

    @classmethod
    def from_dict(
        cls,
        spec: dict[str, Any],
        *,
        default_selector: Selector | None = None,
        default_crossover: Crossover | None = None,
        default_mutator: Mutator | None = None,
        default_terminator: Terminator | None = None,
    ) -> "Specification":
        selectors = spec.get("selector", dict())
        crossovers = spec.get("crossover", dict())
        mutators = spec.get("mutator", dict())
        terminators = spec.get("terminator", dict())

        return cls(
            selectors=SelectorFactory(selectors, default_selector),
            crossovers=CrossoverFactory(crossovers, default_crossover),
            mutators=MutatorFactory(mutators, default_mutator),
            terminators=TerminatorFactory(terminators, default_terminator),
        )

    @classmethod
    def parse_toml(
        cls,
        fp: os.PathLike,
        *,
        default_selector: Selector | None = None,
        default_crossover: Crossover | None = None,
        default_mutator: Mutator | None = None,
        default_terminator: Terminator | None = None,
    ) -> "Specification":
        with open(fp, "rb") as f:
            spec = tomllib.load(f)
        return cls.from_dict(
            spec,
            default_selector=default_selector,
            default_crossover=default_crossover,
            default_mutator=default_mutator,
            default_terminator=default_terminator,
        )
