from random import randint, shuffle
from typing import ClassVar

from sonin.model.neuron import TetanicPeriod


class Mutagen:
    def __init__(self, occurrence_weight: int | None = None, deviation_weight: int | None = None):
        self.occurrence_weight: int = occurrence_weight or 1
        self.deviation_weight: int = deviation_weight or 1

        assert self.occurrence_weight >= 1
        assert self.deviation_weight >= 1

    def mutate(self, num_mutations: int):
        raise NotImplementedError(f"{self.__class__.__name__}.mutate")


class Mutator:
    def __init__(self, mutagens: list[Mutagen]):
        self.mutagens: list[Mutagen] = mutagens
        self.mutagen_map = {i: m for i, m in enumerate(self.mutagens)}
        self.mutation_selector = [i for i, m in self.mutagen_map.items() for _ in range(m.occurrence_weight)]

    def mutate(self, num_mutations: int | None = None):
        assert num_mutations > 0

        num_mutations = num_mutations or 1
        shuffle(self.mutation_selector)
        mutations = {}

        for i in range(num_mutations):
            selection = self.mutation_selector[i % len(self.mutation_selector)]

            if selection in mutations:
                mutations[selection] += 1
            else:
                mutations[selection] = 1

        for i, num in mutations.items():
            self.mutagen_map[i].mutate(num)


class IntMutagen(Mutagen):
    MIN: ClassVar[int] = -(2 ** 63)
    MAX: ClassVar[int] = 2 ** 63 - 1

    def __init__(
        self,
        value: int,
        min_value: int | None = None,
        max_value: int | None = None,
        occurrence_weight: int | None = None,
        deviation_weight: int | None = None,
    ):
        super().__init__(
            occurrence_weight=occurrence_weight,
            deviation_weight=deviation_weight,
        )

        self._value: int = value
        self.min_value: int = max(min_value or self.MIN, self.MIN)
        self.max_value: int = min(max_value or self.MAX, self.MAX)

        assert self.MIN <= self.min_value <= self.value <= self.max_value <= self.MAX

    @property
    def value(self) -> int:
        return self._value

    @value.setter
    def value(self, value: int):
        self._value = max(self.min_value, min(self.max_value, value))

    def mutate(self, num_mutations: int):
        sign = randint(0, 1) * 2 - 1
        deviation = randint(1, self.deviation_weight * num_mutations)
        self.value += sign * deviation


class UintMutagen(IntMutagen):
    MIN: ClassVar[int] = 0
    MAX: ClassVar[int] = 2 ** 64 - 1


class TetanicPeriodMutagen(Mutagen):
    def __init__(
        self,
        threshold: UintMutagen,
        activations: UintMutagen,
        gap: UintMutagen,
        occurrence_weight: int | None = None,
        deviation_weight: int | None = None,
    ):
        super().__init__(
            occurrence_weight=occurrence_weight,
            deviation_weight=deviation_weight,
        )

        self._threshold: UintMutagen = threshold
        self._activations: UintMutagen = activations
        self._gap: UintMutagen = gap
        self._mutator: Mutator = Mutator([threshold, activations, gap])

    @property
    def value(self) -> int:
        return TetanicPeriod()

    def mutate(self, num_mutations: int):
        self._mutator.mutate(num_mutations)
