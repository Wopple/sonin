import heapq
import time
from datetime import timedelta
from itertools import groupby
from math import prod
from typing import Self

from pydantic import BaseModel

from sonin.model.dna import Dna
from sonin.model.gear import Gear
from sonin.model.lesson import Lesson, LessonPlan
from sonin.model.metric import Metric
from sonin.model.mind import MindInterface
from sonin.model.mind_factory import MindFactory
from sonin.model.mutation import Mutator
from sonin.model.step import HasStep
from sonin.sonin_math import div, most_significant_bit
from sonin.sonin_random import HasRandom, Pcg32, rand_int, Random


def in_tolerance(b, m) -> bool:
    return b <= m <= b + 1 + most_significant_bit(b)


class Sample(BaseModel):
    dna: Dna
    measurements: tuple[int, ...] | None = None
    baselines: tuple[int, ...] | None = None
    tolerances: tuple[int, ...] | None = None
    health_mask: tuple[bool, ...] | None = None
    paint_count_metric: Metric | None = None

    @property
    def total_fitness(self) -> tuple[int, int]:
        # parent-relative noise snap: if a measurement is within the parent's tolerance
        # band, treat it as the parent's value so noise doesn't propagate
        snapped = (
            tuple(
                b if in_tolerance(b, m) else m
                for b, m in zip(self.baselines, self.measurements)
            )
            if self.baselines is not None
            else self.measurements
        )

        # if mask not set yet, treat every axis as health (gen-0 semantics)
        mask = self.health_mask or (True,) * len(snapped)
        # if no bands yet, every axis has band=0 (gen-0 / pre-tolerance semantics)
        tols = self.tolerances or (0,) * len(snapped)

        # primary key: product of (1 + excess) over health axes only.
        # axes inside their band contribute factor 1 (multiplicative identity).
        health = prod(
            1 + max(0, m - t)
            for m, t, h in zip(snapped, tols, mask)
            if h
        )
        # secondary key: product of task measurements
        task = prod(m for m, h in zip(snapped, mask) if not h)

        return health, task

    def build_next(
        self,
        dna: Dna,
        tolerances: tuple[int, ...] | None = None,
        health_mask: tuple[bool, ...] | None = None,
    ) -> Self:
        return Sample(
            dna=dna,
            baselines=self.measurements,
            tolerances=tolerances,
            health_mask=health_mask,
        )

    def __eq__(self, other: Self) -> bool:
        return self.total_fitness == other.total_fitness

    def __lt__(self, other: Self) -> bool:
        return self.total_fitness < other.total_fitness

    def __le__(self, other: Self) -> bool:
        return self.total_fitness <= other.total_fitness

    def __gt__(self, other: Self) -> bool:
        return self.total_fitness > other.total_fitness

    def __ge__(self, other: Self) -> bool:
        return self.total_fitness >= other.total_fitness


class Coach(HasStep):
    """
    Interacts with a mind to elicit behavior.
    Rewards good behavior and punishes bad behavior.
    Records metrics for the purpose of producing a final measurement of fitness.
    """

    # Required on every concrete Coach. 'health' axes participate in tolerance bands;
    # 'task' axes are evaluated only after health is satisfied.
    kind: str
    measurement_arity: int

    def __init__(self):
        self._mind: MindInterface | None = None
        self._sample: Sample | None = None
        self.done: bool = False

    @property
    def health_mask(self) -> tuple[bool, ...]:
        return (self.kind == 'health',) * self.measurement_arity

    @property
    def mind(self) -> MindInterface | None:
        return self._mind

    @mind.setter
    def mind(self, mind: MindInterface):
        self._mind = mind

    @property
    def sample(self) -> Sample | None:
        return self._sample

    @sample.setter
    def sample(self, sample: Sample):
        self._sample = sample

    @property
    def num_dimensions(self) -> int:
        return self.mind.mind.num_dimensions

    @property
    def dimension_size(self) -> int:
        return self.mind.mind.dimension_size

    @property
    def num_neurons(self) -> int:
        return len(self.mind.mind.neurons.items)

    def step(self, c_time: int):
        self.pre_step(c_time)
        self.mind.step(c_time)
        self.post_step(c_time)

    def cleanup(self, c_time: int):
        self.mind.cleanup(c_time)

    # useful to send input
    def pre_step(self, c_time: int):
        pass

    # useful to read output and accumulate performance metrics
    def post_step(self, c_time: int):
        pass

    def measure(self) -> tuple[tuple[int, ...], LessonPlan]:
        raise NotImplementedError(f'{self.__class__.__name__}.measure')

    def reset(self):
        self.done = False


class Coaches(Coach):
    def __init__(self, coaches: list[Coach]):
        Coach.__init__(self)
        self.coaches = coaches

    @property
    def measurement_arity(self) -> int:
        return sum(c.measurement_arity for c in self.coaches)

    @property
    def health_mask(self) -> tuple[bool, ...]:
        mask: tuple[bool, ...] = ()
        for c in self.coaches:
            mask += c.health_mask
        return mask

    @property
    def mind(self) -> MindInterface | None:
        return self._mind

    @mind.setter
    def mind(self, mind: MindInterface):
        self._mind = mind

        for coach in self.coaches:
            coach.mind = mind

    @property
    def sample(self) -> Sample | None:
        return self._sample

    @sample.setter
    def sample(self, sample: Sample):
        self._sample = sample

        for coach in self.coaches:
            coach.sample = sample

    def step(self, c_time: int):
        for coach in self.coaches:
            if not coach.done:
                coach.pre_step(c_time)

        self.mind.step(c_time)

        for coach in self.coaches:
            if not coach.done:
                coach.post_step(c_time)

        self.done = all(coach.done for coach in self.coaches)

    def measure(self) -> tuple[tuple[int, ...], LessonPlan]:
        measurements = ()
        lesson_plan = LessonPlan(plan={})

        for coach in self.coaches:
            next_measurements, next_lesson_plan = coach.measure()
            measurements += next_measurements
            lesson_plan += next_lesson_plan

        return measurements, lesson_plan

    def reset(self):
        Coach.reset(self)

        for coach in self.coaches:
            coach.reset()


class Health(Coach):
    """
    Selects for traits indicative of a healthy mind.
    """

    kind = 'health'
    measurement_arity = 8

    def __init__(
        self,
        n_time: int = 0,
        d_time: int = 64,
    ):
        Coach.__init__(self)

        # time of the next input
        self.n_time = n_time

        # time when done
        self.d_time = d_time

        self.over_activations: int = 0
        self.under_activations: int = 0
        self.activation_variance_miss: int = 0

        # used to incentivise variation in the activations otherwise we get simple oscillation
        self.activations_set_counts: dict[tuple[int, ...], int] = {}
        self.previous_activations_set: tuple[int, ...] | None = None

    @property
    def target_axon_load(self) -> int:
        return self.dimension_size

    def post_step(self, c_time: int):
        # target partial activity to avoid too much and too little activity
        target_activations = div(self.num_neurons, 4)
        num_activations = self.mind.mind.num_activations

        if num_activations > target_activations:
            self.over_activations += num_activations - target_activations
        elif num_activations < target_activations:
            self.under_activations += target_activations - num_activations

        # record the set of activations to avoid repetition
        activations_set = tuple(self.mind.mind.activation_set)

        if activations_set in self.activations_set_counts:
            self.activations_set_counts[activations_set] += 1
        else:
            self.activations_set_counts[activations_set] = 1

        if self.previous_activations_set is not None:
            self.activation_variance_miss += sum(
                bin(a & b).count('1')
                for a, b in zip(activations_set, self.previous_activations_set)
            )

        self.previous_activations_set = activations_set

        if c_time >= self.d_time:
            self.done = True

    def measure(self) -> tuple[tuple[int, ...], LessonPlan]:
        # target activations
        target_activations_component = div(self.over_activations + self.under_activations, self.d_time) + 1

        # activation variance
        activation_variance_component = div(self.activation_variance_miss, self.d_time) + 1

        # axon distance
        axons_over_distance = 0
        axons_under_distance = 0

        for n in self.mind.mind.neurons:
            distance_diff = self.mind.mind.dimension_size - n.position.city_distance(n.axon.position)

            if distance_diff > 0:
                axons_over_distance += distance_diff
            elif distance_diff < 0:
                axons_under_distance += -distance_diff

        target_axon_distance_component = div(axons_over_distance + axons_under_distance, self.num_neurons) + 1

        # axon load
        max_axons_in_same_position = max(
            len(list(g))
            for _, g in groupby(sorted(n.axon.position.value for n in self.mind.mind.neurons))
        )

        axon_load_component = abs(self.target_axon_load - max_axons_in_same_position) + 1

        # axon variance
        relative_values: dict[tuple[int, ...], int] = {}

        for n in self.mind.mind.neurons:
            relative_value = (n.position - n.axon.position).value

            if relative_value in relative_values:
                relative_values[relative_value] += 1
            else:
                relative_values[relative_value] = 1

        axon_variance_component = max(relative_values.values())

        # activations set
        activations_set_component = max(self.activations_set_counts.values())

        # paint mean
        paint_mean_portion = div(self.num_neurons, 8)
        clipped_mean = min(paint_mean_portion, div(self.num_neurons, self.sample.paint_count_metric.mean))
        paint_mean_component = paint_mean_portion - clipped_mean + 1

        # paint instability
        if self.sample.paint_count_metric.size >= div(paint_mean_portion, 2):
            # only put this component into play once there are sufficient paints
            average_instability = div(self.sample.paint_count_metric.instability, self.sample.paint_count_metric.size)
            paint_instability_portion = min(self.num_neurons, average_instability + 1)
        else:
            # otherwise max it out
            paint_instability_portion = self.num_neurons

        measurements = (
            target_activations_component,
            activation_variance_component,
            target_axon_distance_component,
            axon_load_component,
            axon_variance_component,
            activations_set_component,
            paint_mean_component,
            paint_instability_portion,
        )

        need_more_axon_movement = False
        need_less_axon_movement = False

        if target_axon_distance_component > min(measurements) * 2:
            if axons_over_distance >= axons_under_distance:
                need_more_axon_movement = True
            else:
                need_less_axon_movement = True

        plan = {
            Lesson.MORE_AXON_MOVEMENT: Gear(up=target_axon_distance_component if need_more_axon_movement else 1),
            Lesson.LESS_AXON_MOVEMENT: Gear(up=target_axon_distance_component if need_less_axon_movement else 1),
        }

        return measurements, LessonPlan(plan=plan)

    def reset(self):
        Coach.reset(self)
        self.n_time = 0
        self.over_activations = 0
        self.under_activations = 0
        self.activation_variance_miss = 0
        self.activations_set_counts = {}
        self.previous_activations_set = None


class Echo(Coach):
    kind = 'task'
    measurement_arity = 1

    def __init__(self):
        Coach.__init__(self)

        # expect these values to be echoed back
        # dict[time, value]
        self.expectations: dict[int, int] = {}

        # time when done
        self.d_time: int = 64

        # expect values to be echoed back in this many steps
        self.delay: int = 2

        # number of times the output did not match the expectation
        self.output_misses: int = 0

    @property
    def max_input(self) -> int:
        return 2 ** len(self.mind.input_neurons) - 1

    def pre_step(self, c_time: int):
        value = rand_int(0, self.max_input)
        self.expectations[c_time + self.delay] = value
        self.mind.input(c_time, value)

    def post_step(self, c_time: int):
        output = self.mind.output()
        expected = self.expectations.get(c_time, 0)

        if output != expected:
            self.output_misses += bin(output ^ expected).count('1')

        if c_time >= self.d_time:
            self.done = True

    def measure(self) -> tuple[tuple[int, ...], LessonPlan]:
        return (self.output_misses + 1,), LessonPlan(plan={})

    def reset(self):
        Coach.reset(self)
        self.output_misses = 0


class PetriDish(HasRandom):
    """
    Evolves DNA selecting the best fitness.
    """

    def __init__(
        self,
        coach: Coach,
        sample_retention: int = 4,
        num_descendants: int = 4,
        num_mutations: int = 4,
        health_patience: int = 64,
    ):
        # exercises the mind while measuring performance
        self.coach = coach

        # how many samples to keep from each generation
        self.sample_retention = sample_retention

        # number of descendants per sample in each generation
        self.num_descendants = num_descendants

        # number of mutations in each descendant
        self.num_mutations = num_mutations

        # generations of no per-axis improvement before health bands freeze
        self.health_patience = health_patience

        self.samples: list[tuple[Sample, LessonPlan]] = []
        self.mutator = Mutator()

        # per-axis health tolerance bands. None while health is still improving;
        # frozen at the per-axis best (plus msb-noise) once plateau is reached.
        self.tolerances: tuple[int, ...] | None = None

        # best (smallest) measurement seen on each axis across all generations
        self.best_measurements: tuple[int, ...] | None = None

        # generations since any axis's best improved
        self.generations_without_improvement: int = 0

    def evolve(
        self,
        initial_samples: list[Sample],
        min_generations: int = 1,
        min_elapsed_time: timedelta = timedelta(seconds=0),
    ):
        start_time = time.time()
        descendants: list[Sample] = initial_samples
        num_generations = 0

        # which measurement axes are health vs task — fixed for the run
        health_mask = self.coach.health_mask

        # tag initial samples so their first total_fitness uses the right mask
        for d in descendants:
            d.health_mask = health_mask

        while num_generations < min_generations or (time.time() - start_time) < min_elapsed_time.total_seconds():
            new_samples: list[tuple[Sample, LessonPlan]] = []

            # the condition ensures we measure the input samples if they have no measurements
            if all(d.measurements is not None for d in descendants):
                # mutate the DNA
                descendants = []

                for sample, lesson_plan in self.samples:
                    for _ in range(self.num_descendants):
                        descendant = sample.dna.model_copy(deep=True)

                        self.mutator.mutate(
                            dna=descendant,
                            num_mutations=self.num_mutations,
                            lesson_plan=lesson_plan,
                        )

                        descendants.append(sample.build_next(
                            descendant,
                            tolerances=self.tolerances,
                            health_mask=health_mask,
                        ))

            for descendant in descendants:
                # build the descendant mind
                rng = Pcg32()
                rng.seed(1)
                factory = MindFactory(descendant.dna)
                mind: MindInterface = factory.build_mind(Random(rng))
                descendant.paint_count_metric = factory.paint_count_metric
                mind.mind.randomize_potential()

                # measure the fitness
                self.coach.mind = mind
                self.coach.sample = descendant
                self.coach.reset()
                c_time = 0

                while not self.coach.done:
                    self.coach.step(c_time)
                    self.coach.cleanup(c_time)
                    c_time += 1

                measurements, lesson_plan = self.coach.measure()
                descendant.measurements = measurements
                new_samples.append((descendant, lesson_plan))

            # refresh retained samples to the current bands so they're ranked under
            # the same tolerances as the new descendants
            for sample, _ in self.samples:
                sample.tolerances = self.tolerances
                sample.health_mask = health_mask

            # keep the most fit
            # prepending new samples allows for drift on ties due to stable sort
            self.samples: list[tuple[Sample, LessonPlan]] = heapq.nsmallest(
                self.sample_retention,
                new_samples + self.samples,
                key=lambda x: x[0],
            )

            # track the per-axis best across retained samples and freeze bands
            # only after `health_patience` generations of no improvement on any
            # axis. Until then, tolerances stay None so health drives selection.
            if self.samples and self.tolerances is None:
                arity = len(self.samples[0][0].measurements)
                gen_best = tuple(
                    min(s.measurements[i] for s, _ in self.samples)
                    for i in range(arity)
                )

                if self.best_measurements is None:
                    self.best_measurements = gen_best
                    self.generations_without_improvement = 0
                else:
                    improved = any(g < b for g, b in zip(gen_best, self.best_measurements))
                    self.best_measurements = tuple(
                        min(g, b) for g, b in zip(gen_best, self.best_measurements)
                    )
                    if improved:
                        self.generations_without_improvement = 0
                    else:
                        self.generations_without_improvement += 1

                if self.generations_without_improvement >= self.health_patience:
                    # freeze bands at best + msb-noise slack so small regressions
                    # that opportunistically buy task progress are tolerated
                    self.tolerances = tuple(
                        b + 1 + most_significant_bit(b)
                        for b in self.best_measurements
                    )

            print((
                num_generations,
                self.generations_without_improvement,
                self.tolerances,
                [(sample.total_fitness, sample.measurements) for sample, _ in self.samples],
            ))

            num_generations += 1
