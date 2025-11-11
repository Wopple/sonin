from typing import Any

from sonin.model.dna import Dna
from sonin.model.fate import BinaryFate, Fate, IsLeft
from sonin.model.lesson import Lesson, LessonPlan
from sonin.model.neuron import TetanicPeriod
from sonin.model.signal import Signal, SignalCount
from sonin.model.stimulation import SnapBack, Stimulation
from sonin.sonin_math import div
from sonin.sonin_random import HasRandom, rand_int

# allows for scaling up and scaling down, 1 would not allow for scaling down while remaining above 0
BASE_WEIGHT = 16

# influences the total number of signals
SIGNALS_WEIGHT = 4


class Mutagen(HasRandom):
    def __init__(self, deviation_weight: int = 1):
        assert deviation_weight >= 1

        self.deviation_weight = deviation_weight

    def mutate(
        self,
        dna: Dna | None = None,
        num_mutations: int = 1,
        lesson_plan: LessonPlan | None = None,
        subject: Any | None = None,
    ):
        raise NotImplementedError(f'{self.__class__.__name__}.mutate')


class IntOps(HasRandom):
    deviation_weight: int = 1

    def __init__(
        self,
        min_value: int = -(2 ** 63),
        max_value: int = 2 ** 63 - 1,
    ):
        self.min_value = min_value
        self.max_value = max_value

    def clip(self, value: int) -> int:
        return max(self.min_value, min(self.max_value, value))

    def add(self, value: int) -> int:
        return self.clip(value + self.rand_int(1, self.deviation_weight))

    def sub(self, value: int) -> int:
        return self.clip(value - self.rand_int(1, self.deviation_weight))

    def update_int(self, value: int) -> int:
        if self.rand_bool():
            return self.add(value)
        else:
            return self.sub(value)


class DimensionSizeMutagen(Mutagen, IntOps):
    def __init__(self, deviation_weight: int = 1):
        Mutagen.__init__(self, deviation_weight=deviation_weight)
        IntOps.__init__(self, min_value=5, max_value=10)

    def mutate(
        self,
        dna: Dna | None = None,
        num_mutations: int = 1,
        lesson_plan: LessonPlan | None = None,
        subject: Any | None = None,
    ):
        dna.dimension_size = self.update_int(dna.dimension_size)


class MaxSynapsesMutagen(Mutagen, IntOps):
    def __init__(self, deviation_weight: int = 1):
        Mutagen.__init__(self, deviation_weight=deviation_weight)
        IntOps.__init__(self, min_value=1, max_value=10)

    def mutate(
        self,
        dna: Dna | None = None,
        num_mutations: int = 1,
        lesson_plan: LessonPlan | None = None,
        subject: Any | None = None,
    ):
        dna.max_synapses = self.update_int(dna.max_synapses)


class MaxSynapseStrengthMutagen(Mutagen, IntOps):
    def __init__(self, deviation_weight: int = 1):
        Mutagen.__init__(self, deviation_weight=deviation_weight)
        IntOps.__init__(self, min_value=1)

    def mutate(
        self,
        dna: Dna | None = None,
        num_mutations: int = 1,
        lesson_plan: LessonPlan | None = None,
        subject: Any | None = None,
    ):
        dna.max_synapse_strength = self.update_int(dna.max_synapse_strength)


class MaxAxonRangeMutagen(Mutagen, IntOps):
    def __init__(self, deviation_weight: int = 1):
        Mutagen.__init__(self, deviation_weight=deviation_weight)
        IntOps.__init__(self, min_value=1)

    def mutate(
        self,
        dna: Dna | None = None,
        num_mutations: int = 1,
        lesson_plan: LessonPlan | None = None,
        subject: Any | None = None,
    ):
        dna.max_axon_range = self.update_int(dna.max_axon_range)


class EnvironmentMutagen(Mutagen):
    def mutate(
        self,
        dna: Dna | None = None,
        num_mutations: int = 1,
        lesson_plan: LessonPlan | None = None,
        subject: Any | None = None,
    ):
        lesson_plan = lesson_plan or LessonPlan(plan={})
        base = BASE_WEIGHT
        add_entry = base
        remove_entry = base if dna.encoded_environment else 0
        change_signal = base if dna.encoded_environment else 0
        move = base if dna.encoded_environment else 0
        inc_signal_count = base if dna.encoded_environment else 0
        dec_signal_count = base if dna.encoded_environment else 0

        weights = [
            (0, lesson_plan[Lesson.MORE_AXON_MOVEMENT](add_entry)),
            (1, lesson_plan[Lesson.LESS_AXON_MOVEMENT](remove_entry)),
            (2, change_signal),
            (3, move),
            (4, lesson_plan[Lesson.MORE_AXON_MOVEMENT](inc_signal_count)),
            (5, lesson_plan[Lesson.LESS_AXON_MOVEMENT](dec_signal_count)),
        ]

        for _ in range(num_mutations):
            match self.weighted_choice(weights):
                case 0: self.add_entry(dna)
                case 1: self.remove_entry(dna)
                case 2: self.change_signal(dna)
                case 3: self.move(dna)
                case 4: self.inc_signal_count(dna)
                case 5: self.dec_signal_count(dna)
                case _: print(f"{self.__class__.__name__} failed to select a mutation")

    def add_entry(self, dna: Dna):
        center = self.rand_int(1, self.deviation_weight)

        dna.encoded_environment.append((
            self.choice(dna.signals),
            [(center, center)] * dna.num_dimensions,
            self.rand_int(1, self.deviation_weight),
        ))

    def remove_entry(self, dna: Dna):
        del dna.encoded_environment[self.rand_int(0, len(dna.encoded_environment) - 1)]

    def change_signal(self, dna: Dna):
        update_idx = self.rand_int(0, len(dna.encoded_environment) - 1)
        _, position, signal_count = dna.encoded_environment[update_idx]
        dna.encoded_environment[update_idx] = self.choice(dna.signals), position, signal_count

    def move(self, dna: Dna):
        update_idx = self.rand_int(0, len(dna.encoded_environment) - 1)
        update_dimension = self.rand_int(0, dna.num_dimensions - 1)
        position = dna.encoded_environment[update_idx][1]
        numerator, delta = position[update_dimension]

        # clip to prevent (0, 0) which would cause a division by 0
        if self.rand_bool():
            clip = 0 if delta > 0 else 1
            new_numerator = max(clip, numerator + self.rand_sign() * self.rand_int(1, self.deviation_weight))
            position[update_dimension] = new_numerator, delta
        else:
            clip = 0 if numerator > 0 else 1
            new_delta = max(clip, delta + self.rand_sign() * self.rand_int(1, self.deviation_weight))
            position[update_dimension] = numerator, new_delta

    def inc_signal_count(self, dna: Dna):
        update_idx = self.rand_int(0, len(dna.encoded_environment) - 1)
        signal, position, signal_count = dna.encoded_environment[update_idx]
        new_count = signal_count + self.rand_int(1, self.deviation_weight)
        dna.encoded_environment[update_idx] = signal, position, new_count

    def dec_signal_count(self, dna: Dna):
        update_idx = self.rand_int(0, len(dna.encoded_environment) - 1)
        signal, position, signal_count = dna.encoded_environment[update_idx]
        deviation = self.rand_int(1, self.deviation_weight)

        if deviation < signal_count:
            dna.encoded_environment[update_idx] = signal, position, signal_count - deviation
        else:
            del dna.encoded_environment[update_idx]


class SignalCountsOps(HasRandom):
    deviation_weight: int

    def add_entry(self, dna: Dna, signal_counts: dict[Signal, SignalCount]):
        for signal in dna.signals:
            if signal not in signal_counts:
                signal_counts[signal] = self.rand_int(1, self.deviation_weight)
                break

    def remove_entry(self, signal_counts: dict[Signal, SignalCount]):
        del signal_counts[self.choice(signal_counts.keys())]

    def change_signal(self, dna: Dna, signal_counts: dict[Signal, SignalCount]):
        update_signal = self.choice(signal_counts.keys())
        candidates = dna.signals - signal_counts.keys()
        new_signal = self.choice(candidates)
        signal_counts[new_signal] = signal_counts[update_signal]
        del signal_counts[update_signal]

    def inc_signal_count(self, signal_counts: dict[Signal, SignalCount]):
        update_signal = self.choice(signal_counts.keys())
        signal_counts[update_signal] += self.rand_int(1, self.deviation_weight)

    def dec_signal_count(self, signal_counts: dict[Signal, SignalCount], allow_delete: bool = True):
        update_signal = self.choice(signal_counts.keys())
        deviation = self.rand_int(1, self.deviation_weight)

        if deviation < signal_counts[update_signal]:
            signal_counts[update_signal] -= deviation
        elif allow_delete:
            del signal_counts[update_signal]
        else:
            signal_counts[update_signal] = 1


class SignalsMutagen(Mutagen):
    def mutate(
        self,
        dna: Dna | None = None,
        num_mutations: int = 1,
        lesson_plan: LessonPlan | None = None,
        subject: Any | None = None,
    ):
        # only adding signals since removing a signal would require removing it in many places
        if dna.signals:
            new_signal = max(dna.signals) + 1
        else:
            new_signal = 0

        dna.signals.add(new_signal)

        # an added signal must be used in order for it to have any effect

        # include it in the environment
        center = self.rand_int(1, self.deviation_weight)

        dna.encoded_environment.append((
            new_signal,
            [(center, center)] * dna.num_dimensions,
            self.rand_int(1, self.deviation_weight),
        ))

        # include it in the incubation signals
        dna.incubation_signals[new_signal] = self.rand_int(1, self.deviation_weight)

        # assign affinities
        if dna.affinities:
            target_signal = self.choice(dna.affinities.keys())
            affinity = self.rand_sign() * self.rand_int(1, self.deviation_weight)
            dna.affinities[target_signal].update({new_signal: affinity})

        source_signal = div(new_signal * self.rand_int(0, len(dna.signals)), len(dna.signals))
        affinity = self.rand_sign() * self.rand_int(1, self.deviation_weight)
        dna.affinities[new_signal] = {source_signal: affinity}


class IncubationSignalsMutagen(Mutagen, SignalCountsOps):
    def mutate(
        self,
        dna: Dna | None = None,
        num_mutations: int = 1,
        lesson_plan: LessonPlan | None = None,
        subject: Any | None = None,
    ):
        base = BASE_WEIGHT
        add_entry = base if len(dna.incubation_signals) < len(dna.signals) else 0
        remove_entry = base if dna.incubation_signals else 0
        change_signal = base if dna.incubation_signals and len(dna.incubation_signals) < len(dna.signals) else 0
        inc_signal_count = base if dna.incubation_signals else 0
        dec_signal_count = base if dna.incubation_signals else 0

        weights = [
            (0, add_entry),
            (1, remove_entry),
            (2, change_signal),
            (3, inc_signal_count),
            (4, dec_signal_count),
        ]

        for _ in range(num_mutations):
            match self.weighted_choice(weights):
                case 0: self.add_entry(dna, dna.incubation_signals)
                case 1: self.remove_entry(dna.incubation_signals)
                case 2: self.change_signal(dna, dna.incubation_signals)
                case 3: self.inc_signal_count(dna.incubation_signals)
                case 4: self.dec_signal_count(dna.incubation_signals)


class SignalProfileMutagen(Mutagen):
    def mutate(
        self,
        dna: Dna | None = None,
        num_mutations: int = 1,
        lesson_plan: LessonPlan | None = None,
        subject: Any | None = None,
    ):
        base = BASE_WEIGHT
        add_target = base if len(dna.affinities) < len(dna.signals) else 0
        remove_target = base if dna.affinities else 0
        change_target = base if dna.affinities and len(dna.affinities) < len(dna.signals) else 0

        if any(len(affinity) < len(dna.signals) for affinity in dna.affinities.values()):
            add_source = base
        else:
            add_source = 0

        remove_source = base if dna.affinities else 0
        update_source_signal_count = base if dna.affinities else 0

        weights = [
            (0, add_target),
            (1, remove_target),
            (2, change_target),
            (3, add_source),
            (4, remove_source),
            (5, update_source_signal_count),
        ]

        for _ in range(num_mutations):
            match self.weighted_choice(weights):
                case 0: self.add_target(dna)
                case 1: self.remove_target(dna)
                case 2: self.change_target(dna)
                case 3: self.add_source(dna)
                case 4: self.remove_source(dna)
                case 5: self.update_source_signal_count(dna)
                case _: print(f"{self.__class__.__name__} failed to select a mutation")

    def add_target(self, dna: Dna):
        new_target = self.choice(dna.signals - dna.affinities.keys())
        new_source = self.choice(dna.signals)
        dna.affinities[new_target] = {new_source: self.rand_sign() * self.rand_int(1, self.deviation_weight)}

    def remove_target(self, dna: Dna):
        del dna.affinities[self.choice(dna.affinities.keys())]

    def change_target(self, dna: Dna):
        update_target = self.choice(dna.affinities.keys())
        new_target = self.choice(dna.signals - dna.affinities.keys())
        dna.affinities[new_target] = dna.affinities[update_target]
        del dna.affinities[update_target]

    def add_source(self, dna: Dna):
        update_sources = self.choice(
            affinity
            for affinity in dna.affinities.values()
            if len(affinity) < len(dna.signals)
        )

        new_source = self.choice(dna.signals - update_sources.keys())
        update_sources[new_source] = self.rand_sign() * self.rand_int(1, self.deviation_weight)

    def remove_source(self, dna: Dna):
        update_target = self.choice(dna.affinities.keys())
        update_sources = dna.affinities[update_target]

        if len(update_sources) > 1:
            # remove a random source if there will still be some sources
            del update_sources[self.choice(update_sources.keys())]
        else:
            # remove the target if it would otherwise end up with no sources
            del dna.affinities[update_target]

    def update_source_signal_count(self, dna: Dna):
        update_target = self.choice(dna.affinities.keys())
        update_sources = dna.affinities[update_target]
        update_source = self.choice(update_sources.keys())
        deviation = self.rand_sign() * self.rand_int(1, self.deviation_weight)

        if deviation != -update_sources[update_source]:
            update_sources[update_source] += deviation
        elif len(update_sources) > 1:
            # remove a random source if there will still be some sources
            del update_sources[update_source]
        else:
            # remove the target if it would otherwise end up with no sources
            del dna.affinities[update_target]


class AxonSignalsMutagen(Mutagen, SignalCountsOps):
    def mutate(
        self,
        dna: Dna | None = None,
        num_mutations: int = 1,
        lesson_plan: LessonPlan | None = None,
        subject: Any | None = None,
    ):
        assert isinstance(subject, Fate)

        base = BASE_WEIGHT
        add_entry = base if len(subject.axon_signals) < len(dna.signals) else 0
        remove_entry = base if len(subject.axon_signals) > 1 else 0
        change_signal = base if subject.axon_signals and len(subject.axon_signals) < len(dna.signals) else 0
        inc_signal_count = base if subject.axon_signals else 0
        dec_signal_count = base if subject.axon_signals else 0

        weights = [
            (0, add_entry),
            (1, remove_entry),
            (2, change_signal),
            (3, inc_signal_count),
            (4, dec_signal_count),
        ]

        match self.weighted_choice(weights):
            case 0: self.add_entry(dna, subject.axon_signals)
            case 1: self.remove_entry(subject.axon_signals)
            case 2: self.change_signal(dna, subject.axon_signals)
            case 3: self.inc_signal_count(subject.axon_signals)
            case 4: self.dec_signal_count(subject.axon_signals, allow_delete=False)
            case _: print(f"{self.__class__.__name__} failed to select a mutation")


class StimulationMutagen(Mutagen, SignalCountsOps):
    def __init__(
        self,
        deviation_weight: int = 1,
        amount: IntOps | None = None,
        restore_rate: IntOps | None = None,
        restore_damper: IntOps | None = None,
    ):
        Mutagen.__init__(self, deviation_weight=deviation_weight)
        self.amount = amount or IntOps(min_value=1)
        self.restore_rate = restore_rate or IntOps(min_value=1)
        self.restore_damper = restore_damper or IntOps(min_value=0)

    def mutate(
        self,
        dna: Dna | None = None,
        num_mutations: int = 1,
        lesson_plan: LessonPlan | None = None,
        subject: Any | None = None,
    ):
        assert isinstance(subject, Fate)

        base = BASE_WEIGHT
        inc_amount = base
        dec_amount = base
        inc_restore_rate = base
        dec_restore_rate = base
        inc_restore_damper = base
        dec_restore_damper = base

        weights = [
            (0, inc_amount),
            (1, dec_amount),
            (2, inc_restore_rate),
            (3, dec_restore_rate),
            (4, inc_restore_damper),
            (5, dec_restore_damper),
        ]

        match self.weighted_choice(weights):
            case 0: self.inc_amount(subject.stimulation)
            case 1: self.dec_amount(subject.stimulation)
            case 2: self.inc_restore_rate(subject.stimulation)
            case 3: self.dec_restore_rate(subject.stimulation)
            case 4: self.inc_restore_damper(subject.stimulation)
            case 5: self.dec_restore_damper(subject.stimulation)
            case _: print(f"{self.__class__.__name__} failed to select a mutation")

    def inc_amount(self, stimulation: Stimulation):
        stimulation.amount = self.amount.add(stimulation.amount)

    def dec_amount(self, stimulation: Stimulation):
        stimulation.amount = self.amount.sub(stimulation.amount)

    def inc_restore_rate(self, stimulation: Stimulation):
        stimulation.snap_back.restore_rate = self.restore_rate.add(stimulation.snap_back.restore_rate)

    def dec_restore_rate(self, stimulation: Stimulation):
        new_rate = self.restore_rate.sub(stimulation.snap_back.restore_rate)
        stimulation.snap_back.restore_rate = max(new_rate, stimulation.snap_back.restore_damper)

    def inc_restore_damper(self, stimulation: Stimulation):
        new_rate = self.restore_damper.add(stimulation.snap_back.restore_damper)
        stimulation.snap_back.restore_damper = min(new_rate, stimulation.snap_back.restore_rate)

    def dec_restore_damper(self, stimulation: Stimulation):
        stimulation.snap_back.restore_damper = self.restore_damper.sub(stimulation.snap_back.restore_damper)


class TetanicPeriodMutagen(Mutagen, SignalCountsOps):
    def __init__(
        self,
        deviation_weight: int = 1,
        threshold: IntOps | None = None,
        activations: IntOps | None = None,
        gap: IntOps | None = None,
    ):
        Mutagen.__init__(self, deviation_weight=deviation_weight)
        self.threshold = threshold or IntOps(min_value=0)
        self.activations = activations or IntOps(min_value=1)
        self.gap = gap or IntOps(min_value=0)

    def mutate(
        self,
        dna: Dna | None = None,
        num_mutations: int = 1,
        lesson_plan: LessonPlan | None = None,
        subject: Any | None = None,
    ):
        assert isinstance(subject, Fate)

        base = BASE_WEIGHT
        flip_enabled = base
        inc_threshold = base
        dec_threshold = base
        inc_activations = base
        dec_activations = base
        inc_gap = base
        dec_gap = base

        weights = [
            (0, flip_enabled),
            (1, inc_threshold),
            (2, dec_threshold),
            (3, inc_activations),
            (4, dec_activations),
            (5, inc_gap),
            (6, dec_gap),
        ]

        match self.weighted_choice(weights):
            case 0: self.flip_enabled(subject.tetanic_period)
            case 1: self.inc_threshold(subject.tetanic_period)
            case 2: self.dec_threshold(subject.tetanic_period)
            case 3: self.inc_activations(subject.tetanic_period)
            case 4: self.dec_activations(subject.tetanic_period)
            case 5: self.inc_gap(subject.tetanic_period)
            case 6: self.dec_gap(subject.tetanic_period)
            case _: print(f"{self.__class__.__name__} failed to select a mutation")

    def flip_enabled(self, tetanic_period: TetanicPeriod):
        tetanic_period.enabled = not tetanic_period.enabled

    def inc_threshold(self, tetanic_period: TetanicPeriod):
        tetanic_period.threshold = self.threshold.add(tetanic_period.threshold)

    def dec_threshold(self, tetanic_period: TetanicPeriod):
        tetanic_period.threshold = self.threshold.sub(tetanic_period.threshold)

    def inc_activations(self, tetanic_period: TetanicPeriod):
        tetanic_period.activations = self.activations.add(tetanic_period.activations)

    def dec_activations(self, tetanic_period: TetanicPeriod):
        tetanic_period.activations = self.activations.add(tetanic_period.activations)

    def inc_gap(self, tetanic_period: TetanicPeriod):
        tetanic_period.gap = self.gap.sub(tetanic_period.gap)

    def dec_gap(self, tetanic_period: TetanicPeriod):
        tetanic_period.gap = self.gap.sub(tetanic_period.gap)


class FateMutagen(Mutagen, SignalCountsOps):
    def __init__(
        self,
        deviation_weight: int = 1,
        axon_signals: AxonSignalsMutagen | None = None,
        activation_level: IntOps | None = None,
        refactory_period: IntOps | None = None,
        stimulation: StimulationMutagen | None = None,
        overstimulation_threshold: IntOps | None = None,
        tetanic_period: TetanicPeriodMutagen | None = None,
    ):
        Mutagen.__init__(self, deviation_weight=deviation_weight)
        self.axon_signals = axon_signals or AxonSignalsMutagen()
        self.activation_level = activation_level or IntOps(min_value=1)
        self.refactory_period = refactory_period or IntOps(min_value=0)
        self.stimulation = stimulation or StimulationMutagen()
        self.overstimulation_threshold = overstimulation_threshold or IntOps(min_value=1)
        self.tetanic_period = tetanic_period or TetanicPeriodMutagen()

    def mutate(
        self,
        dna: Dna | None = None,
        num_mutations: int = 1,
        lesson_plan: LessonPlan | None = None,
        subject: Any | None = None,
    ):
        assert isinstance(subject, Fate)

        base = BASE_WEIGHT
        flip_excites = base
        update_axon_signals = base if dna.signals else 0
        update_activation_level = base
        update_refactory_period = base
        update_stimulation = base
        update_overstimulation_threshold = base
        update_tetanic_period = base

        weights = [
            (0, flip_excites),
            (1, update_axon_signals),
            (2, update_activation_level),
            (3, update_refactory_period),
            (4, update_stimulation),
            (5, update_overstimulation_threshold),
            (6, update_tetanic_period),
        ]

        for _ in range(num_mutations):
            match self.weighted_choice(weights):
                case 0: self.flip_excites(subject)
                case 1: self.axon_signals.mutate(dna, subject=subject)
                case 2: self.update_activation_level(subject)
                case 3: self.update_refactory_period(subject)
                case 4: self.stimulation.mutate(subject=subject)
                case 5: self.update_overstimulation_threshold(subject)
                case 6: self.tetanic_period.mutate(subject=subject)
                case _: print(f"{self.__class__.__name__} failed to select a mutation")

    def flip_excites(self, fate: Fate):
        fate.excites = not fate.excites

    def update_activation_level(self, fate: Fate):
        fate.activation_level = self.activation_level.update_int(fate.activation_level)

    def update_refactory_period(self, fate: Fate):
        fate.refactory_period = self.activation_level.update_int(fate.refactory_period)

    def update_overstimulation_threshold(self, fate: Fate):
        fate.overstimulation_threshold = self.activation_level.update_int(fate.overstimulation_threshold)


class BinaryFateMutagen(Mutagen):
    def mutate(
        self,
        dna: Dna | None = None,
        num_mutations: int = 1,
        lesson_plan: LessonPlan | None = None,
        subject: Any | None = None,
    ):
        assert isinstance(subject, BinaryFate)

        base = BASE_WEIGHT
        add_entry = base if len(subject.is_left) < 2 * len(dna.signals) else 0
        remove_entry = base if len(subject.is_left) > 1 else 0
        change_signal = base if subject.is_left else 0
        flip_is_lower = base if subject.is_left else 0
        inc_threshold = base if subject.is_left else 0
        dec_threshold = base if subject.is_left else 0

        weights = [
            (0, add_entry),
            (1, remove_entry),
            (2, change_signal),
            (3, flip_is_lower),
            (4, inc_threshold),
            (5, dec_threshold),
        ]

        for _ in range(num_mutations):
            match self.weighted_choice(weights):
                case 0: self.add_entry(dna, subject.is_left)
                case 1: self.remove_entry(subject.is_left)
                case 2: self.change_signal(dna, subject.is_left)
                case 3: self.flip_is_lower(subject.is_left)
                case 4: self.inc_threshold(subject.is_left)
                case 5: self.dec_threshold(subject.is_left)
                case _: print(f"{self.__class__.__name__} failed to select a mutation")

    def add_entry(self, dna: Dna, is_left: IsLeft):
        all_candidates = {(signal, is_lower) for signal in dna.signals for is_lower in (False, True)}
        remaining_candidates = all_candidates - {(signal, is_lower) for signal, is_lower, _ in is_left}
        selected_signal, selected_is_lower = self.choice(remaining_candidates)

        # if the same signal already exists, ensure the new constraint is compatible with the existing constraint
        for signal, is_lower, threshold in is_left:
            if signal == selected_signal:
                if is_lower:
                    # the new threshold must be <= the existing one
                    selected_threshold = self.rand_int(0, threshold)
                else:
                    # the new threshold must be >= the existing one
                    selected_threshold = self.rand_int(threshold, threshold + self.deviation_weight)

                break
        else:
            # the same signal does not exist
            selected_threshold = self.rand_int(0, self.deviation_weight)

        is_left.append((selected_signal, selected_is_lower, selected_threshold))

    def remove_entry(self, is_left: IsLeft):
        del is_left[self.rand_int(0, len(is_left) - 1)]

    def change_signal(self, dna: Dna, is_left: IsLeft):
        update_idx = self.rand_int(0, len(is_left) - 1)
        update_signal, update_is_lower, update_threshold = is_left[update_idx]
        new_signal = self.choice(dna.signals - {update_signal})
        is_left[update_idx] = new_signal, update_is_lower, update_threshold
        deletions: list[int] = []

        # 1. delete the existing signal with the same is_lower if the new constraint is more restrictive
        #      otherwise, delete the row to be updated since it is superfluous
        #    ---|>------<|--- start
        #    ---|>--|>--<|--- add new constraint
        #    -------|>--<|--- drop unnecessary constraint
        #
        #    ---|>------<|--- start
        #    |>-|>------<|--- add new constraint
        #    ---|>------<|--- new constraint was the unnecessary one
        #
        # 2. delete the existing signal with the opposite is_lower if the new constraint is incompatible
        #    ---|>------<|--- start
        #    ---|>------<|-|> add new constraint
        #    --------------|> drop both unnecessary constraints
        if update_is_lower:
            for idx, (signal, is_lower, threshold) in enumerate(is_left):
                if idx != update_idx and signal == new_signal:
                    if threshold > update_threshold:
                        deletions.append(idx)
                    elif is_lower is update_is_lower:
                        deletions.append(update_idx)
                        break
        else:
            for idx, (signal, is_lower, threshold) in enumerate(is_left):
                if idx != update_idx and signal == new_signal:
                    if threshold < update_threshold:
                        deletions.append(idx)
                    elif is_lower is update_is_lower:
                        deletions.append(update_idx)
                        break

        deletions.sort(reverse=True)

        for idx in deletions:
            del is_left[idx]

    def flip_is_lower(self, is_left: IsLeft):
        update_idx = self.rand_int(0, len(is_left) - 1)
        update_signal, update_is_lower, update_threshold = is_left[update_idx]
        is_left[update_idx] = update_signal, not update_is_lower, update_threshold

        # flipping one makes the other constraint on the same signal superfluous
        # ---|>------<|---
        # ---|>-------|>--
        # ------------|>--
        for idx, (signal, _, _) in enumerate(is_left):
            if signal == update_signal and idx != update_idx:
                del is_left[idx]
                break
            

    def inc_threshold(self, is_left: IsLeft):
        update_idx = self.rand_int(0, len(is_left) - 1)
        signal, is_lower, threshold = is_left[update_idx]
        is_left[update_idx] = signal, is_lower, threshold + self.rand_int(1, self.deviation_weight)

    def dec_threshold(self, is_left: IsLeft):
        update_idx = self.rand_int(0, len(is_left) - 1)
        update_signal, update_is_lower, update_threshold = is_left[update_idx]
        deviation = self.rand_int(1, self.deviation_weight)

        # <= in this case because 0 is valid, but -1 is not
        if deviation <= update_threshold:
            is_left[update_idx] = update_signal, update_is_lower, update_threshold - deviation
        else:
            del is_left[update_idx]


class FateTreeMutagen(Mutagen):
    def __init__(
        self,
        deviation_weight: int = 1,
        fate: FateMutagen | None = None,
        binary_fate: BinaryFateMutagen | None = None,
    ):
        Mutagen.__init__(self, deviation_weight=deviation_weight)
        self.fate = fate or FateMutagen()
        self.binary_fate = binary_fate or BinaryFateMutagen()

    def mutate(
        self,
        dna: Dna | None = None,
        num_mutations: int = 1,
        lesson_plan: LessonPlan | None = None,
        subject: Any | None = None,
    ):
        base = BASE_WEIGHT
        add_fate = base if dna.signals else 0
        remove_fate = base if len(tuple(iter(dna.fate_tree))) > 1 else 0
        update_fate = base
        update_binary_fate = base if len(tuple(dna.fate_tree.branches_iter())) > 0 else 0

        weights = [
            (0, add_fate),
            (1, remove_fate),
            (2, update_fate),
            (3, update_binary_fate),
        ]

        for _ in range(num_mutations):
            match self.weighted_choice(weights):
                case 0: self.add_fate(dna)
                case 1: self.remove_fate(dna)
                case 2: self.update_fate(dna)
                case 3: self.update_binary_fate(dna)
                case _: print(f"{self.__class__.__name__} failed to select a mutation")

    def add_fate(self, dna: Dna):
        dna.fate_tree.add(
            leaf=Fate(
                excites=True,
                axon_signals={self.choice(dna.signals): rand_int(1, self.deviation_weight)},
                activation_level=rand_int(1, self.deviation_weight),
                refactory_period=rand_int(0, self.deviation_weight),
                stimulation=Stimulation(
                    amount=rand_int(1, self.deviation_weight),
                    snap_back=SnapBack(
                        baseline=0,
                        restore_rate=rand_int(1, self.deviation_weight),
                        restore_damper=1,
                    ),
                ),
                overstimulation_threshold=rand_int(1, self.deviation_weight),
                tetanic_period=TetanicPeriod(
                    enabled=self.rand_bool(),
                    threshold=rand_int(0, self.deviation_weight),
                    activations=rand_int(1, self.deviation_weight),
                    gap=rand_int(0, self.deviation_weight),
                ),
            ),
            new_branch=lambda left, right: BinaryFate(left=left, right=right, is_left=[(
                self.choice(dna.signals),
                self.rand_bool(),
                rand_int(0, self.deviation_weight),
            )]),
            is_next_left=self.rand_bool,
        )

    def remove_fate(self, dna: Dna):
        dna.fate_tree.remove(self.rand_bool)

    def update_fate(self, dna: Dna):
        self.fate.mutate(dna, subject=self.choice(iter(dna.fate_tree)))

    def update_binary_fate(self, dna: Dna):
        self.binary_fate.mutate(dna, subject=self.choice(dna.fate_tree.branches_iter()))


class Mutator(Mutagen):
    def __init__(
        self,
        deviation_weight: int = 1,
        dimension_size: DimensionSizeMutagen | None = None,
        max_synapses: MaxSynapsesMutagen | None = None,
        max_synapse_strength: MaxSynapseStrengthMutagen | None = None,
        max_axon_range: MaxAxonRangeMutagen | None = None,
        signals: SignalsMutagen | None = None,
        environment: EnvironmentMutagen | None = None,
        incubation_signals: IncubationSignalsMutagen | None = None,
        signal_profile: SignalProfileMutagen | None = None,
        fate_tree: FateTreeMutagen | None = None,
    ):
        Mutagen.__init__(self, deviation_weight=deviation_weight)
        self.dimension_size = dimension_size or DimensionSizeMutagen()
        self.max_synapses = max_synapses or MaxSynapsesMutagen()
        self.max_synapse_strength = max_synapse_strength or MaxSynapseStrengthMutagen()
        self.max_axon_range = max_axon_range or MaxAxonRangeMutagen()
        self.signals = signals or SignalsMutagen()
        self.environment = environment or EnvironmentMutagen()
        self.incubation_signals = incubation_signals or IncubationSignalsMutagen()
        self.signal_profile = signal_profile or SignalProfileMutagen()
        self.fate_tree = fate_tree or FateTreeMutagen()

    def mutate(
        self,
        dna: Dna | None = None,
        num_mutations: int = 1,
        lesson_plan: LessonPlan | None = None,
        subject: Any | None = None,
    ):
        base = BASE_WEIGHT
        dimension_size = 0  # not supporting changes in dimension size yet
        max_synapses = base
        max_synapse_strength = base
        max_axon_range = base

        # since:
        # - signals are important
        # - this only adds signals and does not remove them
        #
        # begins with a high weight and the weight rapidly decreases to 1 as signals are added
        signals = div(base * SIGNALS_WEIGHT + len(dna.signals) + 1, len(dna.signals) + 1)

        environment = base if dna.signals else 0
        incubation_signals = base if dna.signals else 0
        signal_profile = base if dna.signals else 0
        fate_tree = base

        weights = [
            (self.dimension_size, dimension_size),
            (self.max_synapses, max_synapses),
            (self.max_synapse_strength, max_synapse_strength),
            (self.max_axon_range, max_axon_range),
            (self.signals, signals),
            (self.environment, environment),
            (self.incubation_signals, incubation_signals),
            (self.signal_profile, signal_profile),
            (self.fate_tree, fate_tree),
        ]

        for _ in range(num_mutations):
            self.weighted_choice(weights).mutate(dna, lesson_plan=lesson_plan)
