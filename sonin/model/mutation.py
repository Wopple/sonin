from typing import Any

from sonin.model.dna import Dna
from sonin.model.fate import Fate
from sonin.model.hypercube import AbsPosition, RelPosition, Vector
from sonin.model.lesson import Lesson, LessonPlan
from sonin.model.neuron import TetanicPeriod
from sonin.model.paint import CityShape, CompleteFill, Fill, FillShape, ModuloFill, OffsetFill, RectangleShape, Shape
from sonin.model.stimulation import SnapBack, Stimulation
from sonin.sonin_math import div
from sonin.sonin_random import HasRandom, Random

# allows for scaling up and scaling down, 1 would not allow for scaling down while remaining above 0
BASE_WEIGHT = 16

MAX_SYNAPSES = 8


def random_relative_coordinate(deviation_weight: int, random: Random) -> tuple[int, int]:
    numerator = random.rand_int(0, deviation_weight)
    delta = random.rand_int(0, deviation_weight)

    if numerator == 0 and delta == 0:
        if random.rand_bool():
            numerator = random.rand_int(1, deviation_weight)
        else:
            delta = random.rand_int(1, deviation_weight)

    return numerator, delta


def random_fate(
    num_dimensions: int,
    dimension_size: int,
    deviation_weight: int,
    random: Random,
) -> Fate:
    return Fate(
        excites=True,
        axon_offset=tuple(
            random.rand_sign() * random.rand_int(1, min(dimension_size - 1, deviation_weight))
            for _ in range(num_dimensions)
        ),
        activation_level=random.rand_int(1, deviation_weight),
        refactory_period=random.rand_int(0, deviation_weight),
        stimulation=Stimulation(
            amount=random.rand_int(1, deviation_weight),
            snap_back=SnapBack(
                baseline=0,
                restore_rate=random.rand_int(1, deviation_weight),
                restore_damper=1,
            ),
        ),
        overstimulation_threshold=random.rand_int(1, deviation_weight),
        tetanic_period=TetanicPeriod(
            enabled=random.rand_bool(),
            threshold=random.rand_int(0, deviation_weight),
            activations=random.rand_int(1, deviation_weight),
            gap=random.rand_int(0, deviation_weight),
        ),
    )


def random_paint(
    num_dimensions: int,
    dimension_size: int,
    deviation_weight: int,
    random: Random,
) -> Shape:
    selected_shape = random.rand_int(0, 2)

    # avoid complete fill, it would cover up everything else
    selected_fill = random.rand_int(0 if selected_shape != 0 else 1, 2)

    if selected_fill == 0:
        fill: Fill = CompleteFill()
    elif selected_fill == 1:
        divisor = random.rand_int(2, dimension_size ** num_dimensions)
        remainder = random.rand_int(0, divisor - 1)
        fill: Fill = ModuloFill(divisor=divisor, remainder=remainder)
    else:
        base = tuple(
            random.rand_int(0, dimension_size - 1)
            for _ in range(num_dimensions)
        )

        offsets = tuple(
            random.rand_int(0, dimension_size - 1)
            for _ in range(num_dimensions)
        )

        fill: Fill = OffsetFill(base=base, offsets=offsets)

    if selected_shape == 0:
        return FillShape(fill=fill)
    elif selected_shape == 1:
        return RectangleShape(
            center=RelPosition(value=[
                random_relative_coordinate(deviation_weight, random)
                for _ in range(num_dimensions)
            ]),
            sizes=tuple(random.rand_int(1, dimension_size) for _ in range(num_dimensions)),
            fill=fill,
            outline=random.rand_bool(),
            wrap=random.rand_bool(),
        )
    else:
        return CityShape(
            center=RelPosition(value=[
                random_relative_coordinate(deviation_weight, random)
                for _ in range(num_dimensions)
            ]),
            size=random.rand_int(1, div(dimension_size + 1, 2)),
            fill=fill,
            outline=random.rand_bool(),
            wrap=random.rand_bool(),
        )


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

    def clip(self, value: int, min_value: int | None = None, max_value: int | None = None) -> int:
        min_value = max(min_value if min_value is not None else self.min_value, self.min_value)
        max_value = min(max_value if max_value is not None else self.max_value, self.max_value)
        return max(min_value, min(max_value, value))

    def add(self, value: int, min_value: int | None = None, max_value: int | None = None) -> int:
        return self.clip(value + self.rand_int(1, self.deviation_weight), min_value, max_value)

    def sub(self, value: int, min_value: int | None = None, max_value: int | None = None) -> int:
        return self.clip(value - self.rand_int(1, self.deviation_weight), min_value, max_value)

    def update_int(self, value: int, min_value: int | None = None, max_value: int | None = None) -> int:
        if self.rand_bool():
            return self.add(value, min_value, max_value)
        else:
            return self.sub(value, min_value, max_value)


class TupleIntOps(HasRandom):
    deviation_weight: int = 1

    def __init__(
        self,
        min_value: int = -(2 ** 63),
        max_value: int = 2 ** 63 - 1,
    ):
        self.int_ops = IntOps(min_value=min_value, max_value=max_value)

        self.int_ops.random = self.random
        self.int_ops.deviation_weight = self.deviation_weight

    def update_tuple_int(
        self,
        value: tuple[int, ...],
        min_value: int | None = None,
        max_value: int | None = None,
    ) -> tuple[int, ...]:
        update_idx = self.rand_int(0, len(value) - 1)

        return tuple(
            v if idx != update_idx else self.int_ops.update_int(
                value[update_idx],
                min_value=min_value,
                max_value=max_value,
            )
            for idx, v in enumerate(value)
        )


class AbsPositionMutagen(Mutagen, TupleIntOps):
    def __init__(self, deviation_weight: int = 1):
        Mutagen.__init__(self, deviation_weight=deviation_weight)
        TupleIntOps.__init__(self)

    def mutate(
        self,
        dna: Dna | None = None,
        num_mutations: int = 1,
        lesson_plan: LessonPlan | None = None,
        subject: Any | None = None,
    ):
        assert isinstance(subject, AbsPosition)

        min_value = 0
        max_value = dna.dimension_size - 1
        subject.value = Vector.of(
            self.update_tuple_int(subject.value.value, min_value=min_value, max_value=max_value),
            dna.dimension_size,
        )


class RelPositionMutagen(Mutagen, IntOps):
    def __init__(self, deviation_weight: int = 1):
        Mutagen.__init__(self, deviation_weight)
        IntOps.__init__(self)

    def mutate(
        self,
        dna: Dna | None = None,
        num_mutations: int = 1,
        lesson_plan: LessonPlan | None = None,
        subject: Any | None = None,
    ):
        assert isinstance(subject, RelPosition)

        base_weight = BASE_WEIGHT
        update_numerator = base_weight
        update_delta = base_weight

        weights = [
            (0, update_numerator),
            (1, update_delta),
        ]

        for _ in range(num_mutations):
            match self.weighted_choice(weights):
                case 0: self.update_numerator(subject)
                case 1: self.update_delta(subject)
                case _: print(f'{self.__class__.__name__} failed to select a mutation')

    def update_numerator(self, position: RelPosition):
        update_idx = self.rand_int(0, len(position.value) - 1)
        numerator, delta = position.value[update_idx]
        min_value = 0 if delta > 0 else 1
        new_numerator = self.update_int(numerator, min_value=min_value)
        position.value[update_idx] = new_numerator, delta

    def update_delta(self, position: RelPosition):
        update_idx = self.rand_int(0, len(position.value) - 1)
        numerator, delta = position.value[update_idx]
        min_value = 0 if numerator > 0 else 1
        new_delta = self.update_int(delta, min_value=min_value)
        position.value[update_idx] = numerator, new_delta


class PositionMutagen(Mutagen):
    def __init__(
        self,
        deviation_weight: int = 1,
        abs_position: AbsPositionMutagen | None = None,
        rel_position: RelPositionMutagen | None = None,
    ):
        Mutagen.__init__(self, deviation_weight=deviation_weight)
        self.abs_position = abs_position or AbsPositionMutagen()
        self.rel_position = rel_position or RelPositionMutagen()

    def mutate(
        self,
        dna: Dna | None = None,
        num_mutations: int = 1,
        lesson_plan: LessonPlan | None = None,
        subject: Any | None = None,
    ):
        if isinstance(subject, AbsPosition):
            self.abs_position.mutate(dna, num_mutations, lesson_plan, subject)
        elif isinstance(subject, RelPosition):
            self.rel_position.mutate(dna, num_mutations, lesson_plan, subject)
        else:
            raise ValueError(f'Expected Position: {subject}')


class MaxSynapsesMutagen(Mutagen, IntOps):
    def __init__(self, deviation_weight: int = 1):
        Mutagen.__init__(self, deviation_weight=deviation_weight)
        IntOps.__init__(self, min_value=1, max_value=MAX_SYNAPSES)

    def mutate(
        self,
        dna: Dna | None = None,
        num_mutations: int = 1,
        lesson_plan: LessonPlan | None = None,
        subject: Any | None = None,
    ):
        base_weight = BASE_WEIGHT
        inc = base_weight if dna.max_synapses < MAX_SYNAPSES else 0
        dec = base_weight if dna.max_synapses > 1 else 0

        weights = [
            (0, lesson_plan[Lesson.MORE_ACTIVATION](inc)),
            (1, lesson_plan[Lesson.LESS_ACTIVATION](dec)),
        ]

        for _ in range(num_mutations):
            match self.weighted_choice(weights):
                case 0:
                    dna.max_synapses = self.add(dna.max_synapses)
                case 1:
                    dna.max_synapses = self.sub(dna.max_synapses)
                case _: print(f'{self.__class__.__name__} failed to select a mutation')


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
        base_weight = BASE_WEIGHT
        inc = base_weight
        dec = base_weight if dna.max_synapse_strength > 1 else 0

        weights = [
            (0, lesson_plan[Lesson.MORE_ACTIVATION](inc)),
            (1, lesson_plan[Lesson.LESS_ACTIVATION](dec)),
        ]

        for _ in range(num_mutations):
            match self.weighted_choice(weights):
                case 0:
                    dna.max_synapse_strength = self.add(dna.max_synapse_strength)
                case 1:
                    dna.max_synapse_strength = self.sub(dna.max_synapse_strength)
                case _: print(f'{self.__class__.__name__} failed to select a mutation')


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
        dna.max_axon_range = self.update_int(dna.max_axon_range, max_value=div(dna.dimension_size, 3))


class StimulationMutagen(Mutagen):
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
        assert isinstance(subject, Stimulation)

        base_weight = BASE_WEIGHT
        inc_amount = base_weight
        dec_amount = base_weight
        inc_restore_rate = base_weight
        dec_restore_rate = base_weight
        inc_restore_damper = base_weight
        dec_restore_damper = base_weight

        weights = [
            (0, lesson_plan[Lesson.LESS_ACTIVATION](inc_amount)),
            (1, lesson_plan[Lesson.MORE_ACTIVATION](dec_amount)),
            (2, lesson_plan[Lesson.MORE_ACTIVATION](inc_restore_rate)),
            (3, lesson_plan[Lesson.LESS_ACTIVATION](dec_restore_rate)),
            (4, lesson_plan[Lesson.LESS_ACTIVATION](inc_restore_damper)),
            (5, lesson_plan[Lesson.MORE_ACTIVATION](dec_restore_damper)),
        ]

        match self.weighted_choice(weights):
            case 0: self.inc_amount(subject)
            case 1: self.dec_amount(subject)
            case 2: self.inc_restore_rate(subject)
            case 3: self.dec_restore_rate(subject)
            case 4: self.inc_restore_damper(subject)
            case 5: self.dec_restore_damper(subject)
            case _: print(f'{self.__class__.__name__} failed to select a mutation')

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


class TetanicPeriodMutagen(Mutagen):
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
        assert isinstance(subject, TetanicPeriod)

        base_weight = BASE_WEIGHT
        flip_enabled = div(base_weight, 4)
        inc_threshold = base_weight
        dec_threshold = base_weight
        inc_activations = base_weight
        dec_activations = base_weight
        inc_gap = base_weight
        dec_gap = base_weight

        weights = [
            (0, flip_enabled),
            (1, lesson_plan[Lesson.LESS_ACTIVATION](inc_threshold)),
            (2, lesson_plan[Lesson.MORE_ACTIVATION](dec_threshold)),
            (3, lesson_plan[Lesson.MORE_ACTIVATION](inc_activations)),
            (4, lesson_plan[Lesson.LESS_ACTIVATION](dec_activations)),
            (5, lesson_plan[Lesson.LESS_ACTIVATION](inc_gap)),
            (6, lesson_plan[Lesson.MORE_ACTIVATION](dec_gap)),
        ]

        match self.weighted_choice(weights):
            case 0: self.flip_enabled(subject)
            case 1: self.inc_threshold(subject)
            case 2: self.dec_threshold(subject)
            case 3: self.inc_activations(subject)
            case 4: self.dec_activations(subject)
            case 5: self.inc_gap(subject)
            case 6: self.dec_gap(subject)
            case _: print(f'{self.__class__.__name__} failed to select a mutation')

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


class ModuloFillMutagen(Mutagen, IntOps):
    def __init__(self, deviation_weight: int = 1):
        Mutagen.__init__(self, deviation_weight=deviation_weight)
        IntOps.__init__(self)

    def mutate(
        self,
        dna: Dna | None = None,
        num_mutations: int = 1,
        lesson_plan: LessonPlan | None = None,
        subject: Any | None = None,
    ):
        assert isinstance(subject, ModuloFill)

        base_weight = BASE_WEIGHT
        divisor = base_weight
        remainder = base_weight

        weights = [
            (0, divisor),
            (1, remainder),
        ]

        for _ in range(num_mutations):
            match self.weighted_choice(weights):
                case 0: self.update_divisor(dna, subject)
                case 1: self.update_remainder(subject)
                case _: print(f'{self.__class__.__name__} failed to select a mutation')

    def update_divisor(self, dna: Dna, fill: ModuloFill):
        min_value = max(2, fill.remainder + 1)
        max_value = dna.num_neurons
        fill.divisor = self.update_int(fill.divisor, min_value=min_value, max_value=max_value)

    def update_remainder(self, fill: ModuloFill):
        min_value = 0
        max_value = fill.divisor - 1
        fill.remainder = self.update_int(fill.remainder, min_value=min_value, max_value=max_value)


class OffsetFillMutagen(Mutagen, IntOps, TupleIntOps):
    def __init__(self, deviation_weight: int = 1):
        Mutagen.__init__(self, deviation_weight=deviation_weight)
        IntOps.__init__(self)
        TupleIntOps.__init__(self)

    def mutate(
        self,
        dna: Dna | None = None,
        num_mutations: int = 1,
        lesson_plan: LessonPlan | None = None,
        subject: Any | None = None,
    ):
        assert isinstance(subject, OffsetFill)

        base_weight = BASE_WEIGHT
        base = base_weight
        offsets = base_weight

        weights = [
            (0, base),
            (1, offsets),
        ]

        for _ in range(num_mutations):
            match self.weighted_choice(weights):
                case 0: self.update_base(dna, subject)
                case 1: self.update_offsets(dna, subject)
                case _: print(f'{self.__class__.__name__} failed to select a mutation')

    def update_base(self, dna: Dna, fill: OffsetFill):
        min_value = 0
        max_value = dna.dimension_size - 1
        fill.base = self.update_tuple_int(fill.base, min_value=min_value, max_value=max_value)

    def update_offsets(self, dna: Dna, fill: OffsetFill):
        min_value = 0
        max_value = dna.dimension_size - 1
        fill.offsets = self.update_tuple_int(fill.offsets, min_value=min_value, max_value=max_value)


class FillMutagen(Mutagen):
    def __init__(
        self,
        deviation_weight: int = 1,
        modulo_fill: ModuloFillMutagen | None = None,
        offset_fill: OffsetFillMutagen | None = None,
    ):
        Mutagen.__init__(self, deviation_weight=deviation_weight)
        self.modulo_fill = modulo_fill or ModuloFillMutagen(deviation_weight=deviation_weight)
        self.offset_fill = offset_fill or OffsetFillMutagen(deviation_weight=deviation_weight)

    def mutate(
        self,
        dna: Dna | None = None,
        num_mutations: int = 1,
        lesson_plan: LessonPlan | None = None,
        subject: Any | None = None,
    ):
        if isinstance(subject, CompleteFill):
            raise RuntimeError(f'should not have mutated with a CompleteFill which has no mutable fields')
        elif isinstance(subject, ModuloFill):
            self.modulo_fill.mutate(dna, num_mutations, lesson_plan, subject)
        elif isinstance(subject, OffsetFill):
            self.offset_fill.mutate(dna, num_mutations, lesson_plan, subject)
        else:
            raise ValueError(f'Expected Fill: {subject}')


class FillShapeMutagen(Mutagen):
    def __init__(
        self,
        deviation_weight: int = 1,
        fill: FillMutagen | None = None,
    ):
        Mutagen.__init__(self, deviation_weight=deviation_weight)
        self.fill = fill or FillMutagen(deviation_weight=deviation_weight)

    def mutate(
        self,
        dna: Dna | None = None,
        num_mutations: int = 1,
        lesson_plan: LessonPlan | None = None,
        subject: Any | None = None,
    ):
        assert isinstance(subject, FillShape)

        base_weight = BASE_WEIGHT
        flip_outline = div(base_weight, 4)
        update_fill = base_weight if not isinstance(subject.fill, CompleteFill) else 0

        weights = [
            (0, flip_outline),
            (1, update_fill),
        ]

        for _ in range(num_mutations):
            match self.weighted_choice(weights):
                case 0: self.flip_outline(subject)
                case 1: self.fill.mutate(dna=dna, lesson_plan=lesson_plan, subject=subject.fill)
                case _: print(f'{self.__class__.__name__} failed to select a mutation')

    def flip_outline(self, subject: FillShape):
        subject.outline = not subject.outline


class RectangleShapeMutagen(Mutagen, TupleIntOps):
    def __init__(
        self,
        deviation_weight: int = 1,
        center: PositionMutagen | None = None,
        fill: FillMutagen | None = None,
    ):
        Mutagen.__init__(self, deviation_weight=deviation_weight)
        TupleIntOps.__init__(self)
        self.center = center or PositionMutagen(deviation_weight=deviation_weight)
        self.fill = fill or FillMutagen(deviation_weight=deviation_weight)

    def mutate(
        self,
        dna: Dna | None = None,
        num_mutations: int = 1,
        lesson_plan: LessonPlan | None = None,
        subject: Any | None = None,
    ):
        assert isinstance(subject, RectangleShape)

        base_weight = BASE_WEIGHT
        update_center = base_weight
        update_sizes = base_weight
        flip_outline = div(base_weight, 4)
        flip_wrap = div(base_weight, 4)
        update_fill = base_weight if not isinstance(subject.fill, CompleteFill) else 0

        weights = [
            (0, update_center),
            (1, update_sizes),
            (2, flip_outline),
            (3, flip_wrap),
            (4, update_fill),
        ]

        for _ in range(num_mutations):
            match self.weighted_choice(weights):
                case 0: self.center.mutate(dna=dna, lesson_plan=lesson_plan, subject=subject.center)
                case 1: self.update_sizes(dna, subject)
                case 2: self.flip_outline(subject)
                case 3: self.flip_wrap(subject)
                case 4: self.fill.mutate(dna=dna, lesson_plan=lesson_plan, subject=subject.fill)
                case _: print(f'{self.__class__.__name__} failed to select a mutation')

    def update_sizes(self, dna: Dna, subject: RectangleShape):
        min_value = 0
        max_value = dna.dimension_size
        subject.sizes = self.update_tuple_int(subject.sizes, min_value=min_value, max_value=max_value)

    def flip_outline(self, subject: RectangleShape):
        subject.outline = not subject.outline

    def flip_wrap(self, subject: RectangleShape):
        subject.wrap = not subject.wrap


class CityShapeMutagen(Mutagen, IntOps):
    def __init__(
        self,
        deviation_weight: int = 1,
        center: PositionMutagen | None = None,
        fill: FillMutagen | None = None,
    ):
        Mutagen.__init__(self, deviation_weight=deviation_weight)
        IntOps.__init__(self)
        self.center = center or PositionMutagen(deviation_weight=deviation_weight)
        self.fill = fill or FillMutagen(deviation_weight=deviation_weight)

    def mutate(
        self,
        dna: Dna | None = None,
        num_mutations: int = 1,
        lesson_plan: LessonPlan | None = None,
        subject: Any | None = None,
    ):
        assert isinstance(subject, CityShape)

        base_weight = BASE_WEIGHT
        update_center = base_weight
        update_size = base_weight
        flip_outline = div(base_weight, 4)
        flip_wrap = div(base_weight, 4)
        update_fill = base_weight if not isinstance(subject.fill, CompleteFill) else 0

        weights = [
            (0, update_center),
            (1, update_size),
            (2, flip_outline),
            (3, flip_wrap),
            (4, update_fill),
        ]

        for _ in range(num_mutations):
            match self.weighted_choice(weights):
                case 0: self.center.mutate(dna=dna, lesson_plan=lesson_plan, subject=subject.center)
                case 1: self.update_size(dna, subject)
                case 2: self.flip_outline(subject)
                case 3: self.flip_wrap(subject)
                case 4: self.fill.mutate(dna=dna, lesson_plan=lesson_plan, subject=subject.fill)
                case _: print(f'{self.__class__.__name__} failed to select a mutation')

    def update_size(self, dna: Dna, subject: CityShape):
        min_value = 1
        max_value = div(dna.dimension_size + 1, 2)
        subject.size = self.update_int(subject.size, min_value=min_value, max_value=max_value)

    def flip_outline(self, subject: CityShape):
        subject.outline = not subject.outline

    def flip_wrap(self, subject: CityShape):
        subject.wrap = not subject.wrap


class ShapeMutagen(Mutagen):
    def __init__(
        self,
        deviation_weight: int = 1,
        fill_shape: FillShapeMutagen | None = None,
        rectangle_shape: RectangleShapeMutagen | None = None,
        city_shape: CityShapeMutagen | None = None,
    ):
        Mutagen.__init__(self, deviation_weight=deviation_weight)
        self.fill_shape = fill_shape or FillShapeMutagen(deviation_weight=deviation_weight)
        self.rectangle_shape = rectangle_shape or RectangleShapeMutagen(deviation_weight=deviation_weight)
        self.city_shape = city_shape or CityShapeMutagen(deviation_weight=deviation_weight)

    def mutate(
        self,
        dna: Dna | None = None,
        num_mutations: int = 1,
        lesson_plan: LessonPlan | None = None,
        subject: Any | None = None,
    ):
        if isinstance(subject, FillShape):
            self.fill_shape.mutate(dna, num_mutations, lesson_plan, subject)
        elif isinstance(subject, RectangleShape):
            self.rectangle_shape.mutate(dna, num_mutations, lesson_plan, subject)
        elif isinstance(subject, CityShape):
            self.city_shape.mutate(dna, num_mutations, lesson_plan, subject)
        else:
            raise ValueError(f'Expected Shape: {subject}')


class FateMutagen(Mutagen, TupleIntOps):
    def __init__(
        self,
        deviation_weight: int = 1,
        activation_level: IntOps | None = None,
        refactory_period: IntOps | None = None,
        stimulation: StimulationMutagen | None = None,
        overstimulation_threshold: IntOps | None = None,
        tetanic_period: TetanicPeriodMutagen | None = None,
    ):
        Mutagen.__init__(self, deviation_weight=deviation_weight)
        TupleIntOps.__init__(self)
        self.activation_level = activation_level or IntOps(min_value=1)
        self.refactory_period = refactory_period or IntOps(min_value=0)
        self.stimulation = stimulation or StimulationMutagen(deviation_weight=deviation_weight)
        self.overstimulation_threshold = overstimulation_threshold or IntOps(min_value=1)
        self.tetanic_period = tetanic_period or TetanicPeriodMutagen(deviation_weight=deviation_weight)

        self.activation_level.deviation_weight = deviation_weight
        self.refactory_period.deviation_weight = deviation_weight
        self.overstimulation_threshold.deviation_weight = deviation_weight

    def mutate(
        self,
        dna: Dna | None = None,
        num_mutations: int = 1,
        lesson_plan: LessonPlan | None = None,
        subject: Any | None = None,
    ):
        assert isinstance(subject, Fate)

        base_weight = BASE_WEIGHT
        flip_excites = div(base_weight, 4)
        update_axon_offset = base_weight
        update_activation_level = base_weight
        update_refactory_period = base_weight
        update_stimulation = base_weight
        update_overstimulation_threshold = base_weight
        update_tetanic_period = base_weight

        weights = [
            (0, flip_excites),
            (1, lesson_plan[(Lesson.MORE_AXON_MOVEMENT, Lesson.LESS_AXON_MOVEMENT)](update_axon_offset)),
            (2, update_activation_level),
            (3, update_refactory_period),
            (4, update_stimulation),
            (5, update_overstimulation_threshold),
            (6, update_tetanic_period),
        ]

        for _ in range(num_mutations):
            match self.weighted_choice(weights):
                case 0: self.flip_excites(subject)
                case 1: self.update_axon_offset(dna, subject)
                case 2: self.update_activation_level(subject)
                case 3: self.update_refactory_period(subject)
                case 4: self.stimulation.mutate(lesson_plan=lesson_plan, subject=subject.stimulation)
                case 5: self.update_overstimulation_threshold(subject)
                case 6: self.tetanic_period.mutate(lesson_plan=lesson_plan, subject=subject.tetanic_period)
                case _: print(f'{self.__class__.__name__} failed to select a mutation')

    def flip_excites(self, fate: Fate):
        fate.excites = not fate.excites

    def update_axon_offset(self, dna: Dna, fate: Fate):
        min_value = -(dna.dimension_size - 1)
        max_value = dna.dimension_size - 1
        fate.axon_offset = self.update_tuple_int(fate.axon_offset, min_value=min_value, max_value=max_value)

    def update_activation_level(self, fate: Fate):
        fate.activation_level = self.activation_level.update_int(fate.activation_level)

    def update_refactory_period(self, fate: Fate):
        fate.refactory_period = self.activation_level.update_int(fate.refactory_period)

    def update_overstimulation_threshold(self, fate: Fate):
        fate.overstimulation_threshold = self.activation_level.update_int(fate.overstimulation_threshold)


class FatePaintsMutagen(Mutagen):
    def __init__(
        self,
        deviation_weight: int = 1,
        paint: ShapeMutagen | None = None,
        fate: FateMutagen | None = None,
    ):
        Mutagen.__init__(self, deviation_weight=deviation_weight)
        self.paint = paint or ShapeMutagen(deviation_weight=deviation_weight)
        self.fate = fate or FateMutagen(deviation_weight=deviation_weight)

    def mutate(
        self,
        dna: Dna | None = None,
        num_mutations: int = 1,
        lesson_plan: LessonPlan | None = None,
        subject: Any | None = None,
    ):
        base_weight = BASE_WEIGHT
        add = base_weight
        remove = base_weight if len(dna.fate_paints) >= 2 else 0
        swap = (base_weight * 2) if len(dna.fate_paints) >= 3 else 0
        update_paint = (base_weight * 4) if len(dna.fate_paints) >= 2 else 0
        update_fate = (base_weight * 4)
        replace_paint = base_weight if len(dna.fate_paints) >= 2 else 0

        weights = [
            (0, add),
            (1, remove),
            (2, swap),
            (3, update_paint),
            (4, lesson_plan[(Lesson.MORE_AXON_MOVEMENT, Lesson.LESS_AXON_MOVEMENT)](update_fate)),
            (5, replace_paint),
        ]

        for _ in range(num_mutations):
            match self.weighted_choice(weights):
                case 0: self.add(dna)
                case 1: self.remove(dna)
                case 2: self.swap(dna)
                case 3: self.update_paint(dna, lesson_plan)
                case 4: self.update_fate(dna, lesson_plan)
                case 5: self.replace_paint(dna)
                case _: print(f'{self.__class__.__name__} failed to select a mutation')

    def add(self, dna: Dna):
        dna.fate_paints.append((
            random_paint(dna.num_dimensions, dna.dimension_size, self.deviation_weight, self.random),
            random_fate(dna.num_dimensions, dna.dimension_size, self.deviation_weight, self.random),
        ))

    def remove(self, dna: Dna):
        del dna.fate_paints[self.rand_int(1, len(dna.fate_paints) - 1)]

    def swap(self, dna: Dna):
        idx_a = self.rand_int(1, len(dna.fate_paints) - 1)
        idx_b = self.rand_int(1, len(dna.fate_paints) - 2)

        # ensure they are not the same
        if idx_b >= idx_a:
            idx_b = idx_b + 1

        a = dna.fate_paints[idx_a]
        dna.fate_paints[idx_a] = dna.fate_paints[idx_b]
        dna.fate_paints[idx_b] = a

    def update_paint(self, dna: Dna, lesson_plan: LessonPlan):
        update_idx = self.rand_int(1, len(dna.fate_paints) - 1)
        paint, fate = dna.fate_paints[update_idx]
        self.paint.mutate(dna, lesson_plan=lesson_plan, subject=paint)

    def update_fate(self, dna: Dna, lesson_plan: LessonPlan):
        update_idx = self.rand_int(0, len(dna.fate_paints) - 1)
        paint, fate = dna.fate_paints[update_idx]
        self.fate.mutate(dna, lesson_plan=lesson_plan, subject=fate)

    def replace_paint(self, dna: Dna):
        update_idx = self.rand_int(1, len(dna.fate_paints) - 1)
        fate = dna.fate_paints[update_idx][1]

        dna.fate_paints[update_idx] = (
            random_paint(dna.num_dimensions, dna.dimension_size, self.deviation_weight, self.random),
            fate,
        )


class Mutator(Mutagen):
    def __init__(
        self,
        deviation_weight: int = 4,
        max_synapses: MaxSynapsesMutagen | None = None,
        max_synapse_strength: MaxSynapseStrengthMutagen | None = None,
        max_axon_range: MaxAxonRangeMutagen | None = None,
        fate_paints: FatePaintsMutagen | None = None,
    ):
        Mutagen.__init__(self, deviation_weight=deviation_weight)
        self.max_synapses = max_synapses or MaxSynapsesMutagen(deviation_weight=1)
        self.max_synapse_strength = max_synapse_strength or MaxSynapseStrengthMutagen(deviation_weight=deviation_weight)
        self.max_axon_range = max_axon_range or MaxAxonRangeMutagen(deviation_weight=1)
        self.fate_paints = fate_paints or FatePaintsMutagen(deviation_weight=deviation_weight)

    def mutate(
        self,
        dna: Dna | None = None,
        num_mutations: int = 1,
        lesson_plan: LessonPlan | None = None,
        subject: Any | None = None,
    ):
        lesson_plan = lesson_plan or LessonPlan(plan={})
        base_weight = BASE_WEIGHT
        max_synapses = base_weight
        max_synapse_strength = base_weight
        max_axon_range = div(base_weight, 4)
        fate_paints = base_weight * 16

        weights = [
            (self.max_synapses, max_synapses),
            (self.max_synapse_strength, max_synapse_strength),
            (self.max_axon_range, max_axon_range),
            (self.fate_paints, fate_paints),
        ]

        for _ in range(num_mutations):
            self.weighted_choice(weights).mutate(dna, lesson_plan=lesson_plan)
