import heapq
import time
from datetime import timedelta

from pydantic import BaseModel, Field

from sonin.model.dna import Dna
from sonin.model.mind import MindInterface
from sonin.model.mutation import DnaMutagen
from sonin.model.step import HasStep
from sonin.sonin_math import div
from sonin.sonin_random import Pcg32, Random

# 0 is maximum fitness, larger numbers have lower fitness
type Fitness = int


class Coach(BaseModel, HasStep):
    """
    Interacts with a mind to elicit behavior.
    Rewards good behavior and punishes bad behavior.
    Records metrics for the purpose of producing a final measurement of fitness.
    """

    mind: MindInterface | None = None
    done: bool = False

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


class PetriDish(BaseModel):
    """
    Evolves DNA selecting the best fitness.
    """
    samples: list[tuple[DnaMutagen, Fitness]] = Field(default_factory=list)

    # exercises the mind while measuring performance
    coach: Coach

    # how many samples to keep from each generation
    sample_retention: int = 4

    # number of descendants per sample in each generation
    num_descendants: int = 4

    # number of mutations in each descendant
    num_mutations: int = 256

    random: Random = Random(rng=Pcg32())

    def evolve(
        self,
        initial_samples: list[DnaMutagen],
        min_generations: int = 1,
        min_elapsed_time: timedelta = timedelta(seconds=0),
    ):
        num_generations = 0
        start_time = time.time()

        while num_generations < min_generations or (time.time() - start_time) < min_elapsed_time.total_seconds():
            if initial_samples:
                descendants = initial_samples
                initial_samples = None
            else:
                descendants = []

                for sample, _ in self.samples:
                    for _ in range(self.num_descendants):
                        # mutate the DNA
                        descendant = sample.model_copy(deep=True)
                        descendant.mutate(self.num_mutations)
                        descendants.append(descendant)

            for descendant in descendants:
                # build the descendant mind
                dna: Dna = descendant.value
                mind: MindInterface = dna.build_mind()
                mind.mind.randomize_potential()

                # measure the fitness
                self.coach.mind = mind
                self.coach.reset()
                c_time = 0

                while not self.coach.done:
                    self.coach.step(c_time)
                    c_time += 1

                self.samples.append((descendant, self.coach.measure()))

            # keep the most fit
            self.samples = heapq.nsmallest(
                self.sample_retention,
                self.samples,
                key=lambda x: x[1],
            )

            print((num_generations, [fitness for _, fitness in self.samples]))
            num_generations += 1


class Health(Coach):
    """
    Selects for traits indicative of a healthy mind.
    """

    # time of the next input
    n_time: int = 0

    input_frequency: int = 8

    # time when done
    d_time: int = 32

    target_activations_miss: int = 0

    # used to incentivise variation in the activations otherwise we get simple oscillation
    activations_set_counts: dict[tuple[int, ...], int] = Field(default_factory=dict)

    def pre_step(self, c_time: int):
        pass
        # if c_time >= self.n_time:
        #     self.mind.input(c_time, 2 ** len(self.mind.input_neurons) - 1)
        #     self.n_time = c_time + self.input_frequency

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

    def measure(self) -> Fitness:
        # target_activations_component = self.target_activations_miss + 1
        activations_set_component = max(self.activations_set_counts.values())
        return activations_set_component

    def reset(self):
        super().reset()
        self.n_time = 0
        self.target_activations_miss = 0
        self.activations_set_counts = {}
