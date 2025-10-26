from random import randint, shuffle
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from sonin.model.fate import Fate
from sonin.model.neuron import TetanicPeriod
from sonin.sonin_random import rand_sign


class Mutagen(BaseModel):
    occurrence_weight: int = Field(default=1, ge=1)
    deviation_weight: int = Field(default=1, ge=1)

    def mutate(self, num_mutations: int):
        raise NotImplementedError(f"{self.__class__.__name__}.mutate")


class Mutator(Mutagen):
    mutagens: list[Mutagen]
    mutagen_map: dict[int, Mutagen] = None
    mutation_selector: list[int] = None

    def model_post_init(self, context: Any, /):
        assert self.mutagen_map is None
        assert self.mutation_selector is None

        self.mutagen_map = {i: m for i, m in enumerate(self.mutagens)}
        self.mutation_selector = [i for i, m in self.mutagen_map.items() for _ in range(m.occurrence_weight)]

    def mutate(self, num_mutations: int):
        assert num_mutations > 0

        # shuffle to make sure we get a proper randomized distribution especially for a small number of mutations
        shuffle(self.mutation_selector)

        mutations = {}

        # distribute mutations among the mutagens based on occurrence weights
        for i in range(num_mutations):
            selection = self.mutation_selector[i % len(self.mutation_selector)]
            mutations[selection] = mutations.get(selection, 0) + 1

        # perform the distributed mutations
        for i, num in mutations.items():
            self.mutagen_map[i].mutate(num)


class BoolMutagen(Mutagen):
    value: bool

    def mutate(self, num_mutations: int):
        self.value = not self.value


class IntMutagen(Mutagen):
    value: int
    min_value: int | None = None
    max_value: int | None = None

    MIN: ClassVar[int] = -(2 ** 63)
    MAX: ClassVar[int] = 2 ** 63 - 1

    def model_post_init(self, context: Any, /):
        self.min_value = max(self.min_value or self.MIN, self.MIN)
        self.max_value = min(self.max_value or self.MAX, self.MAX)
        self.clip_value()

        assert self.MIN <= self.min_value <= self.value <= self.max_value <= self.MAX

    def mutate(self, num_mutations: int):
        sign = rand_sign()
        deviation = randint(1, self.deviation_weight * num_mutations)
        self.value += sign * deviation
        self.clip_value()

    def clip_value(self):
        self.value = max(self.min_value, min(self.max_value, self.value))


class UintMutagen(IntMutagen):
    MIN: ClassVar[int] = 0
    MAX: ClassVar[int] = 2 ** 64 - 1


class TetanicPeriodMutagen(Mutagen):
    is_none: BoolMutagen
    threshold: UintMutagen
    activations: UintMutagen
    gap: UintMutagen
    mutator: Mutator = None

    def model_post_init(self, context: Any, /):
        assert self.mutator is None

        self.mutator = Mutator(mutagens=[self.is_none, self.threshold, self.activations, self.gap])

    @property
    def value(self) -> TetanicPeriod | None:
        if self.is_none.value:
            return None
        else:
            return TetanicPeriod(
                threshold=self.threshold.value,
                activations=self.activations.value,
                gap=self.gap.value,
            )

    def mutate(self, num_mutations: int):
        self.mutator.mutate(num_mutations)


class FateMutagen(Mutagen):
    excites: BoolMutagen
    activation_level: UintMutagen
    refactory_period: UintMutagen
    stimulation_amount: UintMutagen
    stimulation_restore_rate: UintMutagen
    stimulation_restore_damper: UintMutagen
    tetanic_period: TetanicPeriodMutagen
    mutator: Mutator = None

    def model_post_init(self, context: Any, /):
        assert self.mutator is None

        self.mutator = Mutator(mutagens=[
            self.excites,
            self.activation_level,
            self.refactory_period,
            self.stimulation_amount,
            self.stimulation_restore_rate,
            self.stimulation_restore_damper,
            self.tetanic_period,
        ])

    @property
    def value(self) -> Fate:
        return Fate(
            excites=self.excites.value,
            activation_level=self.activation_level.value,
            refactory_period=self.refactory_period.value,
            stimulation_amount=self.stimulation_amount.value,
            stimulation_restore_rate=self.stimulation_restore_rate.value,
            stimulation_restore_damper=self.stimulation_restore_damper.value,
            tetanic_period=self.tetanic_period.value,
        )

    def mutate(self, num_mutations: int):
        self.mutator.mutate(num_mutations)
