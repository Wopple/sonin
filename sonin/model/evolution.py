import heapq
import time
from datetime import timedelta

from sonin.model.dna import Dna
from sonin.model.mind import MindInterface
from sonin.model.mutation2 import Mutator
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

    def step(self, c_time: int):
        self.pre_step(c_time)
        self.mind.step(c_time)
        self.post_step(c_time)

    def pre_step(self, c_time: int):
        pass

    def post_step(self, c_time: int):
        pass

    def measure(self) -> Fitness:
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
        d_time: int = 32,
        target_activations_miss: int = 0,
        activations_set_counts: dict[tuple[int, ...], int] | None = None,
    ):
        Coach.__init__(self, mind=mind)

        # time of the next input
        self.n_time = n_time

        # time when done
        self.d_time = d_time

        self.target_activations_miss = target_activations_miss

        # used to incentivise variation in the activations otherwise we get simple oscillation
        self.activations_set_counts = activations_set_counts or {}

    def post_step(self, c_time: int):
        # target partial activity to avoid too much and too little activity
        target_activations = div(len(self.mind.mind.neurons.items), 4)
        self.target_activations_miss += abs(target_activations - self.mind.mind.num_activations)

        # record the set of activations to avoid repetition
        activations_set = tuple(self.mind.mind.activation_set)

        if activations_set in self.activations_set_counts:
            self.activations_set_counts[activations_set] += 1
        else:
            self.activations_set_counts[activations_set] = 1

        if c_time >= self.d_time:
            self.done = True

    # TODO: build a lesson plan
    def measure(self) -> Fitness:
        target_activations_component = div(self.target_activations_miss, self.d_time) + 1

        target_axon_distance_component = div(sum(
            abs(self.mind.mind.dimension_size - n.position.city_distance(n.axon.position))
            for n in self.mind.mind.neurons
        ), len(self.mind.mind.neurons.items)) + 1

        activations_set_component = max(self.activations_set_counts.values())

        return (
            target_activations_component
            * target_axon_distance_component
            * activations_set_component
            # TODO: punish the same synapses firing consecutively
        )

    def reset(self):
        super().reset()
        self.n_time = 0
        self.target_activations_miss = 0
        self.activations_set_counts = {}


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
        samples: list[tuple[Dna, Fitness]] | None = None,
        mutator: Mutator | None = None,
    ):
        # exercises the mind while measuring performance
        self.coach = coach

        # how many samples to keep from each generation
        self.sample_retention = sample_retention

        # number of descendants per sample in each generation
        self.num_descendants = num_descendants

        # number of mutations in each descendant
        self.num_mutations = num_mutations

        self.samples = samples or []
        self.mutator = mutator or Mutator()

    def evolve(
        self,
        initial_samples: list[Dna],
        min_generations: int = 1,
        min_elapsed_time: timedelta = timedelta(seconds=0),
    ):
        num_generations = 0
        start_time = time.time()

        while num_generations < min_generations or (time.time() - start_time) < min_elapsed_time.total_seconds():
            new_samples: list[tuple[Dna, Fitness]] = []

            if initial_samples:
                descendants = initial_samples
                initial_samples = None
            else:
                # mutate the DNA
                descendants = []

                for sample, _ in self.samples:
                    for _ in range(self.num_descendants):
                        descendant = sample.model_copy(deep=True)

                        self.mutator.mutate(
                            dna=descendant,
                            num_mutations=self.num_mutations,
                            lesson_plan=None,
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
                key=lambda x: x[1],
            )

            print((num_generations, [fitness for _, fitness in self.samples]))
            num_generations += 1
