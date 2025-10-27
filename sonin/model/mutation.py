from itertools import count
from random import choice, randint, shuffle
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from sonin.model.facilitation import Facilitation
from sonin.model.fate import Fate
from sonin.model.neuron import TetanicPeriod
from sonin.model.signal import Signal, SignalProfile
from sonin.model.stimulation import SnapBack, Stimulation
from sonin.sonin_random import rand_sign


class Mutable:
    def mutate(self, num_mutations: int):
        raise NotImplementedError(f"{self.__class__.__name__}.mutate")


class Mutagen[T](BaseModel, Mutable):
    occurrence_weight: int = Field(default=1, ge=1)
    deviation_weight: int = Field(default=1, ge=1)

    @property
    def value(self) -> T:
        raise NotImplementedError(f"{self.__class__.__name__}.value")


class Mutator(BaseModel, Mutable):
    mutagens: list[Mutagen[Any]]
    mutagen_map: dict[int, Mutagen[Any]] = None
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


class BoolMutagen(Mutagen[bool]):
    bool_value: bool

    @property
    def value(self) -> bool:
        return self.bool_value

    def mutate(self, num_mutations: int):
        self.bool_value = not self.bool_value


class IntMutagen(Mutagen[int]):
    int_value: int
    min_value: int | None = None
    max_value: int | None = None

    MIN: ClassVar[int] = -(2 ** 63)
    MAX: ClassVar[int] = 2 ** 63 - 1

    def model_post_init(self, context: Any, /):
        self.min_value = max(self.min_value or self.MIN, self.MIN)
        self.max_value = min(self.max_value or self.MAX, self.MAX)
        self.clip_value()

        assert self.MIN <= self.min_value <= self.int_value <= self.max_value <= self.MAX

    @property
    def value(self) -> int:
        return self.int_value

    def mutate(self, num_mutations: int):
        sign = rand_sign()
        deviation = randint(1, self.deviation_weight * num_mutations)
        self.int_value += sign * deviation
        self.clip_value()

    def clip_value(self):
        self.int_value = max(self.min_value, min(self.max_value, self.int_value))


class UintMutagen(IntMutagen):
    MIN: ClassVar[int] = 0
    MAX: ClassVar[int] = 2 ** 64 - 1


class OptionalMutagen[T](Mutagen[Any | None]):
    mutagen: Mutagen[T]
    exists: BoolMutagen
    mutator: Mutator = None

    def model_post_init(self, context: Any, /):
        assert self.mutator is None

        self.mutator = Mutator(mutagens=[self.mutagen, self.exists])

    @property
    def value(self) -> T | None:
        if self.exists.value:
            return self.mutagen.value
        else:
            return None

    def mutate(self, num_mutations: int):
        self.mutator.mutate(num_mutations)


class SignalValueMutagen(Mutagen[dict[Signal, int]]):
    signal_counts: dict[Signal, int] = Field(default_factory=dict)
    add_old_weight: int = 3
    add_new_weight: int = 1
    sub_weight: int = None

    def model_post_init(self, context: Any, /):
        if self.sub_weight is None:
            # plus 1 to prefer small numbers in the long run
            self.sub_weight = self.add_old_weight + self.add_new_weight + 1

    @property
    def value(self) -> dict[Signal, int]:
        return self.signal_counts.copy()

    def mutate(self, num_mutations: int):
        # these mutagens pick which action to perform per mutation
        add_old = BoolMutagen(bool_value=False, occurrence_weight=self.add_old_weight)
        add_new = BoolMutagen(bool_value=False, occurrence_weight=self.add_new_weight)
        sub = BoolMutagen(bool_value=False, occurrence_weight=self.sub_weight)
        mutator = Mutator(mutagens=[add_old, add_new, sub])

        for _ in range(num_mutations):
            mutator.mutate(1)
            delta = randint(1, self.deviation_weight)

            if not self.signal_counts or add_new.value:
                # add a random new signal preferring small numbers
                add_new.bool_value = False

                for new_signal in count():
                    if new_signal not in self.signal_counts and randint(0, 1) == 0:
                        self.signal_counts[new_signal] = delta
                        break
            elif add_old.value:
                # add to a random existing signal
                add_old.bool_value = False
                signal = choice(tuple(self.signal_counts.keys()))
                self.signal_counts[signal] += delta
            elif sub.value:
                # subtract from a random existing signal
                sub.bool_value = False
                signal = choice(tuple(self.signal_counts.keys()))

                if delta < self.signal_counts[signal]:
                    self.signal_counts[signal] -= delta
                else:
                    del self.signal_counts[signal]


class SignalProfileMutagen(Mutagen[SignalProfile]):
    affinities: dict[Signal, SignalValueMutagen] = Field(default_factory=dict)

    # affinity weights
    add_weight: int = 1
    update_weight: int = 3
    remove_weight: int = None

    # value weights
    add_old_weight: int | None = None
    add_new_weight: int | None = None
    sub_weight: int | None = None

    def model_post_init(self, context: Any, /):
        if self.remove_weight is None:
            # plus 1 to prefer fewer mappings in the long run
            self.remove_weight = self.add_weight + 1

    @property
    def value(self) -> SignalProfile:
        return SignalProfile(affinities={signal: mutagen.value for signal, mutagen in self.affinities.items()})

    def mutate(self, num_mutations: int):
        # these mutagens pick which action to perform per mutation
        add = BoolMutagen(bool_value=False, occurrence_weight=self.add_weight)
        update = BoolMutagen(bool_value=False, occurrence_weight=self.update_weight)
        remove = BoolMutagen(bool_value=False, occurrence_weight=self.remove_weight)
        mutator = Mutator(mutagens=[add, update, remove])

        for _ in range(num_mutations):
            mutator.mutate(1)

            if not self.affinities or add.value:
                # add a random new signal preferring small numbers
                add.bool_value = False

                for new_signal in count():
                    if new_signal not in self.affinities and randint(0, 1) == 0:
                        mutagen = SignalValueMutagen(
                            add_old_weight=self.add_old_weight,
                            add_new_weight=self.add_new_weight,
                            sub_weight=self.sub_weight,
                        )

                        mutagen.mutate(1)
                        self.affinities[new_signal] = mutagen
                        break
            elif update.value:
                # update a random existing signal
                update.bool_value = False
                signal = choice(tuple(self.affinities.keys()))
                self.affinities[signal].mutate(1)

                if not self.affinities[signal].signal_counts:
                    del self.affinities[signal]
            elif remove.value:
                # remove a random existing signal
                remove.bool_value = False
                signal = choice(tuple(self.affinities.keys()))
                del self.affinities[signal]


class FacilitationMutagen(Mutagen[Facilitation]):
    granularity: UintMutagen
    limit: UintMutagen
    mutator: Mutator = None

    def model_post_init(self, context: Any, /):
        assert self.mutator is None

        self.mutator = Mutator(mutagens=[self.granularity, self.limit])

    @property
    def value(self) -> Facilitation:
        return Facilitation(
            granularity=self.granularity.value,
            limit=self.limit.value,
        )

    def mutate(self, num_mutations: int):
        self.mutator.mutate(num_mutations)


class TetanicPeriodMutagen(Mutagen[TetanicPeriod]):
    threshold: UintMutagen
    activations: UintMutagen
    gap: UintMutagen
    mutator: Mutator = None

    def model_post_init(self, context: Any, /):
        assert self.mutator is None

        self.mutator = Mutator(mutagens=[self.threshold, self.activations, self.gap])

    @property
    def value(self) -> TetanicPeriod:
        return TetanicPeriod(
            threshold=self.threshold.value,
            activations=self.activations.value,
            gap=self.gap.value,
        )

    def mutate(self, num_mutations: int):
        self.mutator.mutate(num_mutations)


class SnapBackMutagen(Mutagen[SnapBack]):
    baseline: IntMutagen
    restore_rate: UintMutagen
    restore_damper: UintMutagen
    mutator: Mutator = None

    def model_post_init(self, context: Any, /):
        assert self.mutator is None

        self.mutator = Mutator(mutagens=[self.baseline, self.restore_rate, self.restore_damper])

    @property
    def value(self) -> SnapBack:
        return SnapBack(
            baseline=self.baseline.value,
            restore_rate=self.restore_rate.value,
            restore_damper=self.restore_damper.value,
        )

    def mutate(self, num_mutations: int):
        self.mutator.mutate(num_mutations)


class StimulationMutagen(Mutagen[Stimulation]):
    amount: UintMutagen
    snap_back: SnapBackMutagen
    mutator: Mutator = None

    def model_post_init(self, context: Any, /):
        assert self.mutator is None

        self.mutator = Mutator(mutagens=[self.amount, self.snap_back])

    @property
    def value(self) -> Stimulation:
        return Stimulation(
            amount=self.amount.value,
            snap_back=self.snap_back.value,
        )

    def mutate(self, num_mutations: int):
        self.mutator.mutate(num_mutations)


class FateMutagen(Mutagen):
    excites: BoolMutagen
    axon_signals: SignalValueMutagen
    activation_level: UintMutagen
    refactory_period: UintMutagen
    stimulation: StimulationMutagen
    tetanic_period: OptionalMutagen[TetanicPeriodMutagen]
    mutator: Mutator = None

    def model_post_init(self, context: Any, /):
        assert self.mutator is None

        self.mutator = Mutator(mutagens=[
            self.excites,
            self.axon_signals,
            self.activation_level,
            self.refactory_period,
            self.stimulation,
            self.tetanic_period,
        ])

    @property
    def value(self) -> Fate:
        return Fate(
            excites=self.excites.value,
            axon_signals=self.axon_signals.value,
            activation_level=self.activation_level.value,
            refactory_period=self.refactory_period.value,
            stimulation=self.stimulation.value,
            tetanic_period=self.tetanic_period.value,
        )

    def mutate(self, num_mutations: int):
        self.mutator.mutate(num_mutations)
