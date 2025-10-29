from itertools import count
from typing import Any, Callable, ClassVar, Self

from pydantic import BaseModel, Field

from sonin.model.facilitation import Facilitation
from sonin.model.fate import BinaryFate, Fate, FateTree, IsLeft, IsLower
from sonin.model.neuron import TetanicPeriod
from sonin.model.signal import Signal, SignalProfile
from sonin.model.stimulation import SnapBack, Stimulation
from sonin.sonin_random import choice, rand_bool, rand_int, rand_sign, shuffle
from sonin.tree import BinaryTree


class Mutable:
    def mutate(self, num_mutations: int):
        raise NotImplementedError(f"{self.__class__.__name__}.mutate")


class Mutagen(BaseModel, Mutable):
    occurrence_weight: int = Field(default=1, ge=1)
    deviation_weight: int = Field(default=1, ge=1)

    @property
    def value(self) -> Any:
        raise NotImplementedError(f"{self.__class__.__name__}.value")


class Mutator(BaseModel, Mutable):
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
    bool_value: bool = False

    @property
    def value(self) -> bool:
        return self.bool_value

    def mutate(self, num_mutations: int):
        self.bool_value = not self.bool_value


class IntMutagen(Mutagen):
    int_value: int = 0
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
        deviation = rand_int(1, self.deviation_weight * num_mutations)
        self.int_value += sign * deviation
        self.clip_value()

    def clip_value(self):
        self.int_value = max(self.min_value, min(self.max_value, self.int_value))


class UintMutagen(IntMutagen):
    MIN: ClassVar[int] = 0
    MAX: ClassVar[int] = 2 ** 64 - 1


class OptionalMutagen[T](Mutagen):
    mutagen: Mutagen
    exists: BoolMutagen = Field(default_factory=BoolMutagen)

    @property
    def value(self) -> T | None:
        if self.exists.value:
            return self.mutagen.value
        else:
            return None

    def mutate(self, num_mutations: int):
        if self.exists.value:
            # if it exists, randomly choose the mutagen to mutate
            Mutator(mutagens=[
                self.mutagen,
                self.exists,
            ]).mutate(num_mutations)
        else:
            # if it does not exist, bring it into existence
            self.exists.mutate(1)


class SignalValueMutagen(Mutagen):
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

            if not self.signal_counts or add_new.value:
                # add a random new signal preferring small numbers
                add_new.bool_value = False
                self.signal_counts[self.new_key()] = rand_int(1, self.deviation_weight)
            elif add_old.value:
                # add to a random existing signal
                add_old.bool_value = False
                signal = choice(self.signal_counts.keys())
                self.signal_counts[signal] += rand_int(1, self.deviation_weight)
            elif sub.value:
                # subtract from a random existing signal
                sub.bool_value = False
                signal = choice(self.signal_counts.keys())
                delta = rand_int(1, self.deviation_weight)

                if delta < self.signal_counts[signal]:
                    self.signal_counts[signal] -= delta
                else:
                    del self.signal_counts[signal]

    def new_key(self) -> Signal:
        for new_signal in count():
            if new_signal not in self.signal_counts and rand_bool():
                return new_signal

        raise RuntimeError("unreachable")


class SignalProfileMutagen(Mutagen):
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

                mutagen = SignalValueMutagen(
                    add_old_weight=self.add_old_weight,
                    add_new_weight=self.add_new_weight,
                    sub_weight=self.sub_weight,
                )

                mutagen.mutate(1)
                self.affinities[self.new_key()] = mutagen
            elif update.value:
                # update a random existing signal
                update.bool_value = False
                signal = choice(self.affinities.keys())
                self.affinities[signal].mutate(1)

                if not self.affinities[signal].signal_counts:
                    del self.affinities[signal]
            elif remove.value:
                # remove a random existing signal
                remove.bool_value = False
                signal = choice(self.affinities.keys())
                del self.affinities[signal]

    def new_key(self) -> Signal:
        for new_signal in count():
            if new_signal not in self.affinities and rand_bool():
                return new_signal

        raise RuntimeError("unreachable")


class FacilitationMutagen(Mutagen):
    granularity: UintMutagen
    limit: UintMutagen

    @property
    def value(self) -> Facilitation:
        return Facilitation(
            granularity=self.granularity.value,
            limit=self.limit.value,
        )

    def mutate(self, num_mutations: int):
        Mutator(mutagens=[
            self.granularity,
            self.limit,
        ]).mutate(num_mutations)


class TetanicPeriodMutagen(Mutagen):
    threshold: UintMutagen = Field(default_factory=UintMutagen)

    activations: UintMutagen = Field(default_factory=lambda: UintMutagen(
        int_value=1,
        min_value=1,
    ))

    gap: UintMutagen = Field(default_factory=UintMutagen)

    @property
    def value(self) -> TetanicPeriod:
        return TetanicPeriod(
            threshold=self.threshold.value,
            activations=self.activations.value,
            gap=self.gap.value,
        )

    def mutate(self, num_mutations: int):
        Mutator(mutagens=[
            self.threshold,
            self.activations,
            self.gap,
        ]).mutate(num_mutations)


class SnapBackMutagen(Mutagen):
    baseline: IntMutagen = Field(default_factory=IntMutagen)
    restore_rate_delta: UintMutagen = Field(default_factory=UintMutagen)
    restore_damper: UintMutagen = Field(default_factory=UintMutagen)

    @property
    def value(self) -> SnapBack:
        return SnapBack(
            baseline=self.baseline.value,

            # restore_rate >= restore_damper
            restore_rate=max(self.restore_damper.value + self.restore_rate_delta.value, 1),

            restore_damper=self.restore_damper.value,
        )

    def mutate(self, num_mutations: int):
        Mutator(mutagens=[
            self.baseline,
            self.restore_rate_delta,
            self.restore_damper,
        ]).mutate(num_mutations)


class StimulationMutagen(Mutagen):
    amount: UintMutagen = Field(default_factory=lambda: UintMutagen(
        int_value=1,
        min_value=1,
    ))

    snap_back: SnapBackMutagen = Field(default_factory=SnapBackMutagen)

    @property
    def value(self) -> Stimulation:
        return Stimulation(
            amount=self.amount.value,
            snap_back=self.snap_back.value,
        )

    def mutate(self, num_mutations: int):
        Mutator(mutagens=[
            self.amount,
            self.snap_back,
        ]).mutate(num_mutations)


class FateMutagen(Mutagen):
    excites: BoolMutagen = Field(default_factory=BoolMutagen)
    axon_signals: SignalValueMutagen = Field(default_factory=SignalValueMutagen)

    activation_level: UintMutagen = Field(default_factory=lambda: UintMutagen(
        int_value=1,
        min_value=1,
    ))

    refactory_period: UintMutagen = Field(default_factory=UintMutagen)
    stimulation: StimulationMutagen = Field(default_factory=StimulationMutagen)

    tetanic_period: OptionalMutagen[TetanicPeriodMutagen] = Field(
        default_factory=lambda: OptionalMutagen[TetanicPeriodMutagen](mutagen=TetanicPeriodMutagen())
    )

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
        Mutator(mutagens=[
            self.excites,
            self.axon_signals,
            self.activation_level,
            self.refactory_period,
            self.stimulation,
            self.tetanic_period,
        ]).mutate(num_mutations)


class IsLeftMutagen(Mutagen):
    is_left: IsLeft = Field(default_factory=dict)
    new_weight: int = 1
    remove_weight: int = None
    update_weight: int = 5
    change_signal_weight: int = 1
    is_lower_weight: int = 1
    add_weight: int = 1
    sub_weight: int = 1

    def model_post_init(self, context: Any, /):
        if self.remove_weight is None:
            # prefer fewer constraints in the long run
            self.remove_weight = self.new_weight + 1

    @property
    def value(self) -> IsLeft:
        return self.is_left.copy()

    def mutate(self, num_mutations: int):
        # these mutagens pick which action to perform per mutation
        new = BoolMutagen(bool_value=False, occurrence_weight=self.new_weight)
        remove = BoolMutagen(bool_value=False, occurrence_weight=self.remove_weight)
        update = BoolMutagen(bool_value=False, occurrence_weight=self.update_weight)
        change_signal = BoolMutagen(bool_value=False, occurrence_weight=self.change_signal_weight)
        is_lower = BoolMutagen(bool_value=False, occurrence_weight=self.is_lower_weight)
        add = BoolMutagen(bool_value=False, occurrence_weight=self.add_weight)
        sub = BoolMutagen(bool_value=False, occurrence_weight=self.sub_weight)
        mutator = Mutator(mutagens=[new, remove, update])
        update_mutator = Mutator(mutagens=[change_signal, is_lower, add, sub])

        for _ in range(num_mutations):
            mutator.mutate(1)

            if not self.is_left or new.value:
                # add a random new key preferring small numbers
                new.bool_value = False
                self.is_left[self.new_key()] = rand_int(1, self.deviation_weight)
            elif remove.value:
                # remove a random existing key
                remove.bool_value = False
                del self.is_left[choice(self.is_left.keys())]
            elif update.value:
                # update a random existing key
                update.bool_value = False
                update_mutator.mutate(1)
                update_key = choice(self.is_left.keys())

                if change_signal.value:
                    # change the value the key's signal preferring small numbers
                    change_signal.bool_value = False

                    self.is_left[self.new_key([update_key[1]])] = self.is_left[update_key]
                    del self.is_left[update_key]
                elif is_lower.value:
                    # invert the key's IsLower, also invert the new key's IsLower if it already exists
                    is_lower.bool_value = False
                    new_key = update_key[0], not update_key[1]

                    if new_key in self.is_left:
                        # swap
                        temp = self.is_left[new_key]
                        self.is_left[new_key] = self.is_left[update_key]
                        self.is_left[update_key] = temp
                    else:
                        # move
                        self.is_left[new_key] = self.is_left[update_key]
                        del self.is_left[update_key]
                elif add.value:
                    # increase key's threshold
                    add.bool_value = False
                    self.is_left[update_key] += rand_int(1, self.deviation_weight)
                elif add.value:
                    # decrease key's threshold
                    add.bool_value = False
                    new_threshold = self.is_left[update_key] - rand_int(1, self.deviation_weight)
                    self.is_left[update_key] = max(new_threshold, 0)

    def new_key(self, is_lowers: list[IsLower] | None = None) -> tuple[Signal, IsLower]:
        is_lowers = is_lowers or [False, True]

        for new_signal in count():
            for new_is_lower in is_lowers:
                key = new_signal, new_is_lower

                if key not in self.is_left and rand_bool():
                    return key

        raise RuntimeError("unreachable")


class BinaryFateMutagen(Mutagen):
    left: FateMutagen | Self
    right: FateMutagen | Self
    is_left: IsLeftMutagen = Field(default_factory=IsLeftMutagen)

    @property
    def value(self) -> BinaryFate:
        return BinaryFate(
            left=self.left.value,
            right=self.right.value,
            is_left=self.is_left.value,
        )

    def mutate(self, num_mutations: int):
        Mutator(mutagens=[
            self.left,
            self.right,
            self.is_left,
        ]).mutate(num_mutations)


class FateTreeMutagen(Mutagen, BinaryTree):
    new_weight: int = 1
    remove_weight: int = 1
    update_weight: int = 14
    is_next_left: Callable[[], bool] = rand_bool

    # BinaryTree
    root: FateMutagen | BinaryFateMutagen | None = None
    is_leaf: Callable[[FateMutagen | BinaryFateMutagen], bool] = lambda n: isinstance(n, FateMutagen)

    @property
    def value(self) -> FateTree:
        return FateTree(root=self.root.value)

    def mutate(self, num_mutations: int):
        new = BoolMutagen(bool_value=False, occurrence_weight=self.new_weight)
        remove = BoolMutagen(bool_value=False, occurrence_weight=self.remove_weight)
        update = BoolMutagen(bool_value=False, occurrence_weight=self.update_weight)
        mutator = Mutator(mutagens=[new, remove, update])

        for _ in range(num_mutations):
            mutator.mutate(1)

            if self.root is None or new.value:
                # add a new random fate to the tree
                new.bool_value = False
                self.add(FateMutagen(), lambda l, r: BinaryFateMutagen(left=l, right=r))
            elif remove.value:
                # remove a random fate from the tree
                remove.bool_value = False
                self.remove()
            elif update.value:
                # perform an in-place mutation
                update.bool_value = False
                self.root.mutate(1)
