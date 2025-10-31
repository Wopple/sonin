from itertools import count
from typing import Any, Callable, ClassVar, Self

from pydantic import BaseModel, Field, field_validator

from sonin.model.dna import Dna
from sonin.model.facilitation import Facilitation
from sonin.model.fate import BinaryFate, Fate, FateTree, IsLeft, IsLower
from sonin.model.hypercube import Vector
from sonin.model.neuron import TetanicPeriod
from sonin.model.signal import Signal, SignalCount, SignalProfile
from sonin.model.stimulation import SnapBack, Stimulation
from sonin.sonin_math import div
from sonin.sonin_random import choice, rand_bool, rand_int, rand_sign, shuffle
from sonin.tree import BinaryTree

# list[tuple[Numerator, DenominatorDelta]]
#
# Each item encodes the relative position within the dimension associated with its position in the list. The list must
# be of size n_dimension. Each position component is calculated as:
#
# dimension_size * Numerator // (Numerator + DenominatorDelta)
#
# (1, 1) means: 1 // (1 + 1) or 50%
# (3, 1) means: 3 // (3 + 1) or 75%
# (2, 3) means: 2 // (2 + 3) or 40%
# ((3, 1), (2, 3)) means: 75% along dimension 0 and 40% along dimension 1
#
# This allows position encodings to remain consistent across changes in dimension size.
type Position = tuple[tuple[int, int], ...]


class Mutable:
    def mutate(self, num_mutations: int):
        raise NotImplementedError(f'{self.__class__.__name__}.mutate')


class Mutagen(BaseModel, Mutable):
    occurrence_weight: int = Field(default=1, ge=1)
    deviation_weight: int = Field(default=1, ge=1)

    @property
    def value(self) -> Any:
        raise NotImplementedError(f'{self.__class__.__name__}.value')


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
    new_weight: int = 1
    add_weight: int = 3
    sub_weight: int = None

    def model_post_init(self, context: Any, /):
        if self.sub_weight is None:
            # plus 1 to prefer small numbers in the long run
            self.sub_weight = self.new_weight + self.add_weight + 1

    @property
    def value(self) -> dict[Signal, int]:
        return self.signal_counts.copy()

    def mutate(self, num_mutations: int):
        # these mutagens pick which action to perform per mutation
        new = BoolMutagen(bool_value=False, occurrence_weight=self.new_weight)
        add = BoolMutagen(bool_value=False, occurrence_weight=self.add_weight)
        sub = BoolMutagen(bool_value=False, occurrence_weight=self.sub_weight)
        mutator = Mutator(mutagens=[new, add, sub])

        for _ in range(num_mutations):
            mutator.mutate(1)

            if not self.signal_counts or new.value:
                # add a random new signal preferring small numbers
                new.bool_value = False
                self.signal_counts[self.new_key()] = rand_int(1, self.deviation_weight)
            elif add.value:
                # add to a random existing signal
                add.bool_value = False
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

        raise RuntimeError('unreachable')


class SignalProfileMutagen(Mutagen):
    affinities: dict[Signal, SignalValueMutagen] = Field(default_factory=dict)

    # affinity weights
    add_weight: int = 1
    update_weight: int = 3
    remove_weight: int = None

    # value weights
    signal_new_weight: int = 1
    signal_add_weight: int = 3
    signal_sub_weight: int = None

    def model_post_init(self, context: Any, /):
        if self.remove_weight is None:
            # plus 1 to prefer fewer mappings in the long run
            self.remove_weight = self.add_weight + 1

        if self.signal_sub_weight is None:
            # plus 1 to prefer small numbers in the long run
            self.signal_sub_weight = self.signal_new_weight + self.signal_add_weight + 1

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
                    new_weight=self.signal_new_weight,
                    add_weight=self.signal_add_weight,
                    sub_weight=self.signal_sub_weight,
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

        raise RuntimeError('unreachable')


class FacilitationMutagen(Mutagen):
    granularity: UintMutagen = Field(default_factory=lambda: UintMutagen(int_value=1, min_value=1))
    limit: UintMutagen = Field(default_factory=UintMutagen)

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
    activations: UintMutagen = Field(default_factory=lambda: UintMutagen(int_value=1, min_value=1))
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
    amount: UintMutagen = Field(default_factory=lambda: UintMutagen(int_value=1, min_value=1))
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


class EnvironmentMutagen(Mutagen):
    environment: list[tuple[Signal, Position, SignalCount]] = Field(default_factory=list)
    n_dimension: int
    new_weight: int = 1
    remove_weight: int = 1
    update_weight: int = 14
    change_signal_weight: int = 1
    position_weight: int = 3
    add_count_weight: int = 6
    sub_count_weight: int = 6

    @property
    def value(self) -> dict[tuple[Signal, Position], SignalCount]:
        return {
            (
                signal,
                tuple(
                    (numerator, numerator + delta)
                    for numerator, delta in position
                ),
            ): signal_count
            for signal, position, signal_count in self.environment
        }

    def mutate(self, num_mutations: int):
        new = BoolMutagen(bool_value=False, occurrence_weight=self.new_weight)
        remove = BoolMutagen(bool_value=False, occurrence_weight=self.remove_weight)
        update = BoolMutagen(bool_value=False, occurrence_weight=self.update_weight)
        change_signal = BoolMutagen(bool_value=False, occurrence_weight=self.change_signal_weight)
        position = BoolMutagen(bool_value=False, occurrence_weight=self.position_weight)
        add_count = BoolMutagen(bool_value=False, occurrence_weight=self.add_count_weight)
        sub_count = BoolMutagen(bool_value=False, occurrence_weight=self.sub_count_weight)
        mutator = Mutator(mutagens=[new, remove, update])
        update_mutator = Mutator(mutagens=[change_signal, position, add_count, sub_count])

        for _ in range(num_mutations):
            mutator.mutate(1)

            if not self.environment or new.value:
                # add a new random signal preferring small numbers
                new.bool_value = False
                self.environment.append(self.new_key() + (rand_int(1, self.deviation_weight),))
            elif remove.value:
                # add a random signal
                remove.bool_value = False
                del self.environment[choice(enumerate(self.environment))[0]]
            elif update.value:
                # update a random signal
                update.bool_value = False
                update_mutator.mutate(1)
                update_idx = choice(enumerate(self.environment))[0]

                if change_signal.value:
                    change_signal.bool_value = False
                    self.update_change_signal(update_idx)
                elif position.value:
                    position.bool_value = False
                    self.update_position(update_idx)
                elif add_count.value:
                    add_count.bool_value = False
                    self.update_add_count(update_idx)
                elif sub_count.value:
                    sub_count.bool_value = False
                    self.update_sub_count(update_idx)

    def update_change_signal(self, update_idx: int):
        # change to a new signal
        update_value = self.environment[update_idx]
        new_signal = self.new_signal()

        if new_signal != update_value[0]:
            self.environment.append((new_signal, update_value[1], update_value[2]))
            del self.environment[update_idx]

    def update_position(self, update_idx: int):
        # change the position
        update_value = self.environment[update_idx]
        dim = choice(range(self.n_dimension))
        position = update_value[1]
        dim_position = position[dim]

        if rand_bool():
            # mutate the numerator
            mutator = UintMutagen(int_value=dim_position[0])
            mutator.mutate(1)

            new_position = tuple(
                (mutator.value, dim_position[1]) if idx == dim else p
                for idx, p in enumerate(position)
            )
        else:
            # mutate the denominator delta
            # min_value=1 because (0, 0) results in a division by zero, (0, 1) instead divides by 1
            mutator = UintMutagen(int_value=dim_position[1], min_value=1)
            mutator.mutate(1)

            new_position = tuple(
                (dim_position[0], mutator.value) if idx == dim else p
                for idx, p in enumerate(position)
            )

        existing_idx = None

        for idx, value in enumerate(self.environment):
            if value[0] == update_value[0] and value[1] == new_position:
                existing_idx = idx
                break

        if existing_idx is not None:
            self.environment[existing_idx] = (
                update_value[0],
                new_position,
                update_value[2] + self.environment[existing_idx][2]
            )
        else:
            self.environment.append((update_value[0], new_position, update_value[2]))

        del self.environment[update_idx]

    def update_add_count(self, update_idx: int):
        # increase the signal count
        update_value = self.environment[update_idx]

        self.environment[update_idx] = (
            update_value[0],
            update_value[1],
            update_value[2] + rand_int(1, self.deviation_weight),
        )

    def update_sub_count(self, update_idx: int):
        # decrease the signal count
        update_value = self.environment[update_idx]
        delta = rand_int(1, self.deviation_weight)

        if update_value[2] > delta:
            self.environment[update_idx] = (
                update_value[0],
                update_value[1],
                update_value[2] - delta,
            )
        else:
            del self.environment[update_idx]

    def new_signal(self) -> Signal:
        # We cannot prevent the same key from showing up multiple times so that the signal can show up in multiple
        # places. It is okay if they start in the same spot because they can move independently.
        for new_signal in count():
            if rand_bool():
                return new_signal

        raise RuntimeError('unreachable')

    def new_key(self) -> tuple[Signal, Position]:
        # start in the middle
        return self.new_signal(), tuple((1, 1) for _ in range(self.n_dimension))


class FateMutagen(Mutagen):
    excites: BoolMutagen = Field(default_factory=BoolMutagen)
    axon_signals: SignalValueMutagen = Field(default_factory=SignalValueMutagen)
    activation_level: UintMutagen = Field(default_factory=lambda: UintMutagen(int_value=1, min_value=1))
    refactory_period: UintMutagen = Field(default_factory=UintMutagen)
    stimulation: StimulationMutagen = Field(default_factory=StimulationMutagen)

    tetanic_period: OptionalMutagen[TetanicPeriodMutagen] = Field(
        default_factory=lambda: OptionalMutagen[TetanicPeriodMutagen](mutagen=TetanicPeriodMutagen())
    )

    @field_validator('tetanic_period', mode='before')
    @classmethod
    def parse_tetanic_period(cls, value: dict) -> OptionalMutagen[TetanicPeriodMutagen]:
        return OptionalMutagen[TetanicPeriodMutagen](
            occurrence_weight=value['occurrence_weight'],
            deviation_weight=value['deviation_weight'],
            mutagen=TetanicPeriodMutagen(**value['mutagen']),
            exists=BoolMutagen(**value['exists']),
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
                elif sub.value:
                    # decrease key's threshold
                    sub.bool_value = False
                    new_threshold = self.is_left[update_key] - rand_int(1, self.deviation_weight)
                    self.is_left[update_key] = max(new_threshold, 0)

    def new_key(self, is_lowers: list[IsLower] | None = None) -> tuple[Signal, IsLower]:
        is_lowers = is_lowers or [False, True]

        for new_signal in count():
            for new_is_lower in is_lowers:
                key = new_signal, new_is_lower

                if key not in self.is_left and rand_bool():
                    return key

        raise RuntimeError('unreachable')


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

    # BinaryTree
    root: FateMutagen | BinaryFateMutagen = Field(default_factory=FateMutagen)

    is_leaf: Callable[[FateMutagen | BinaryFateMutagen], bool] = Field(
        default=lambda n: isinstance(n, FateMutagen),
        exclude=True,
    )

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

            if new.value:
                # add a new random fate to the tree
                new.bool_value = False
                self.add(FateMutagen(), lambda l, r: BinaryFateMutagen(left=l, right=r))
            elif remove.value:
                # remove a random fate from the tree
                remove.bool_value = False

                if not self.is_leaf(self.root):
                    self.remove()
            elif update.value:
                # perform an in-place mutation
                update.bool_value = False
                self.root.mutate(1)


class DnaMutagen(Mutagen):
    n_dimension: int = 2
    dimension_size: int = 10
    n_synapse_mutagen: UintMutagen = Field(default_factory=lambda: UintMutagen(int_value=1, min_value=1, max_value=10))
    activation_level_mutagen: UintMutagen = Field(default_factory=lambda: UintMutagen(int_value=1, min_value=1))
    max_neuron_strength_mutagen: UintMutagen = Field(default_factory=lambda: UintMutagen(int_value=1, min_value=1))
    axon_range_mutagen: UintMutagen = Field(default_factory=lambda: UintMutagen(int_value=1, min_value=1, max_value=10))
    refactory_period_mutagen: UintMutagen = Field(default_factory=lambda: UintMutagen(max_value=5))
    environment_mutagen: EnvironmentMutagen = None
    incubation_signals_mutagen: SignalValueMutagen = Field(default_factory=SignalValueMutagen)
    signal_profile_mutagen: SignalProfileMutagen = Field(default_factory=SignalProfileMutagen)
    fate_tree_mutagen: FateTreeMutagen = Field(default_factory=FateTreeMutagen)

    def model_post_init(self, context: Any, /):
        if self.environment_mutagen is None:
            self.environment_mutagen = EnvironmentMutagen(n_dimension=self.n_dimension)

    @property
    def value(self) -> Dna:
        raw_environment = self.environment_mutagen.value

        environment: list[tuple[Signal, SignalCount, Vector]] = [
            (
                signal,
                signal_count,
                Vector.of(
                    [div(self.dimension_size * numerator, numerator + delta) for numerator, delta in position],
                    self.dimension_size,
                ),
            )
            for (signal, position), signal_count in raw_environment.items()
        ]

        return Dna(
            n_dimension=self.n_dimension,
            dimension_size=self.dimension_size,
            n_synapse=self.n_synapse_mutagen.value,
            activation_level=self.activation_level_mutagen.value,
            max_neuron_strength=self.max_neuron_strength_mutagen.value,
            axon_range=self.axon_range_mutagen.value,
            refactory_period=self.refactory_period_mutagen.value,
            environment=environment,
            incubation_signals=self.incubation_signals_mutagen.value,
            signal_profile=self.signal_profile_mutagen.value,
            fate_tree=self.fate_tree_mutagen.value,
        )

    def mutate(self, num_mutations: int):
        Mutator(mutagens=[
            self.n_synapse_mutagen,
            self.activation_level_mutagen,
            self.max_neuron_strength_mutagen,
            self.axon_range_mutagen,
            self.refactory_period_mutagen,
            self.environment_mutagen,
            self.signal_profile_mutagen,
            self.fate_tree_mutagen,
        ]).mutate(num_mutations)
