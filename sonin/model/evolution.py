import heapq
import time
from datetime import timedelta
from itertools import groupby

from sonin.model.dna import Dna
from sonin.model.gear import Gear
from sonin.model.lesson import Lesson, LessonPlan
from sonin.model.mind import MindInterface
from sonin.model.mutation import Mutator
from sonin.model.step import HasStep
from sonin.sonin_math import div
from sonin.sonin_random import HasRandom

# 1 is maximum fitness, larger numbers have lower fitness. 1 is used instead of zero because multiplying by zero
# eliminates all the other fitness criteria.
type Fitness = int


class Coach(HasStep):
    """
    Interacts with a mind to elicit behavior.
    Rewards good behavior and punishes bad behavior.
    Records metrics for the purpose of producing a final measurement of fitness.
    """

    def __init__(self, mind: MindInterface | None = None):
        self.mind = mind
        self.done: bool = False

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

    # useful to send input
    def pre_step(self, c_time: int):
        pass

    # useful to read output and accumulate performance metrics
    def post_step(self, c_time: int):
        pass

    def measure(self) -> tuple[Fitness, LessonPlan]:
        raise NotImplementedError(f'{self.__class__.__name__}.measure')

    def reset(self):
        self.done = False


class Health(Coach):
    """
    Selects for traits indicative of a healthy mind.
    """

    def __init__(
        self,
        mind: MindInterface | None = None,
        n_time: int = 0,
        d_time: int = 64,
    ):
        Coach.__init__(self, mind=mind)

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

    # Approximated by ln(x) / ln(ln(x))
    @property
    def target_axon_load(self) -> int:
        num_neurons = self.num_neurons

        if num_neurons < 743:
            return 3
        elif num_neurons < 41831:
            return 4
        elif num_neurons < 2764920:
            return 5
        else:
            return 6

    def post_step(self, c_time: int):
        # target partial activity to avoid too much and too little activity
        target_activations = div(len(self.mind.mind.neurons.items), 4)
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
                bin(a & b).count("1")
                for a, b in zip(activations_set, self.previous_activations_set)
            )

        self.previous_activations_set = activations_set

        if c_time >= self.d_time:
            self.done = True

    def measure(self) -> tuple[Fitness, LessonPlan]:
        target_activations_component = div(self.over_activations + self.under_activations, self.d_time) + 1
        activation_variance_component = div(self.activation_variance_miss, self.d_time) + 1

        target_axon_distance_component = div(sum(
            abs(self.mind.mind.dimension_size - n.position.city_distance(n.axon.position))
            for n in self.mind.mind.neurons
        ), len(self.mind.mind.neurons.items)) + 1

        max_axons_in_same_position = max(
            len(list(g))
            for _, g in groupby(sorted(n.axon.position.value for n in self.mind.mind.neurons))
        ) + len(self.mind.mind.neurons.items)

        axon_variance_component = abs(self.target_axon_load - max_axons_in_same_position) + 1
        need_more_axon_movement = max_axons_in_same_position < self.target_axon_load
        need_less_axon_movement = max_axons_in_same_position > self.target_axon_load
        activations_set_component = max(self.activations_set_counts.values())

        lesson_weight, lesson = sorted(
            [
                (self.under_activations, Lesson.MORE_ACTIVATION),
                (self.over_activations, Lesson.LESS_ACTIVATION),
                (axon_variance_component if need_more_axon_movement else 1, Lesson.MORE_AXON_MOVEMENT),
                (axon_variance_component if need_less_axon_movement else 1, Lesson.LESS_AXON_MOVEMENT),
            ],
            reverse=True,
        )[0]

        return (
            target_activations_component
            * activation_variance_component
            * target_axon_distance_component
            * axon_variance_component
            * activations_set_component,
            LessonPlan(plan={lesson: Gear(up=lesson_weight)}),
        )

    def reset(self):
        super().reset()
        self.n_time = 0
        self.over_activations = 0
        self.under_activations = 0
        self.activation_variance_miss = 0
        self.activations_set_counts = {}
        self.previous_activations_set = None


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
    ):
        # exercises the mind while measuring performance
        self.coach = coach

        # how many samples to keep from each generation
        self.sample_retention = sample_retention

        # number of descendants per sample in each generation
        self.num_descendants = num_descendants

        # number of mutations in each descendant
        self.num_mutations = num_mutations

        self.samples: list[tuple[Dna, Fitness]] = []
        self.mutator = Mutator()

    def evolve(
        self,
        initial_samples: list[Dna],
        min_generations: int = 1,
        min_elapsed_time: timedelta = timedelta(seconds=0),
    ):
        num_generations = 0
        start_time = time.time()

        while num_generations < min_generations or (time.time() - start_time) < min_elapsed_time.total_seconds():
            new_samples: list[tuple[Dna, tuple[Fitness, LessonPlan]]] = []

            if initial_samples:
                descendants = initial_samples
                initial_samples = None
            else:
                # mutate the DNA
                descendants = []

                for sample, (_, lesson_plan) in self.samples:
                    for _ in range(self.num_descendants):
                        descendant = sample.model_copy(deep=True)

                        self.mutator.mutate(
                            dna=descendant,
                            num_mutations=self.num_mutations,
                            lesson_plan=lesson_plan,
                        )

                        descendants.append(descendant)

            for descendant in descendants:
                # build the descendant mind
                mind: MindInterface = descendant.build_mind()
                mind.mind.randomize_potential()

                # measure the fitness
                self.coach.mind = mind
                self.coach.reset()
                c_time = 0

                while not self.coach.done:
                    self.coach.step(c_time)
                    c_time += 1

                new_samples.append((descendant, self.coach.measure()))

            # keep the most fit
            # prepending new samples allows for drift on ties due to stable sort
            self.samples = heapq.nsmallest(
                self.sample_retention,
                new_samples + self.samples,
                key=lambda x: x[1][0],
            )

            print((num_generations, [fitness for _, (fitness, _) in self.samples]))
            num_generations += 1
