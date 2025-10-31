import heapq
import time
from datetime import timedelta

from pydantic import BaseModel, Field

from sonin.model.dna import Dna
from sonin.model.hypercube import Shape, Vector
from sonin.model.mind import Mind, MindInterface
from sonin.model.mutation import DnaMutagen
from sonin.model.step import HasStep
from sonin.sonin_math import div
from sonin.sonin_random import rand_int

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
        self.input(c_time)
        self.mind.step(c_time)
        self.output(c_time)

    def input(self, c_time: int):
        pass

    def output(self, c_time: int):
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

    # interface definition
    input_shape: Shape
    output_shape: Shape
    reward_shape: Shape
    punish_shape: Shape

    def evolve(
        self,
        initial_sample: DnaMutagen,
        min_generations: int = 1,
        min_elapsed_time: timedelta = timedelta(seconds=0),
    ):
        num_generations = 0
        start_time = time.time()

        while num_generations < min_generations or (time.time() - start_time) < min_elapsed_time.total_seconds():
            descendants = []

            for sample, _ in self.samples:
                for _ in range(self.num_descendants):
                    # mutate the DNA
                    descendant = sample.model_copy(deep=True)
                    descendant.mutate(self.num_mutations)
                    descendants.append(descendant)

            # calculate the fitness of the initial samples
            if num_generations == 0:
                descendants += [s for s, _ in self.samples]
                self.samples = []

            for descendant in descendants:
                # build the descendant mind
                dna: Dna = descendant.value
                mind: Mind = dna.build_mind()

                # configure the interface
                dimension_size = dna.dimension_size
                last_idx = dimension_size - 1
                lower_half_dimension = div(dna.n_dimension, 2)
                upper_half_dimension = dna.n_dimension - lower_half_dimension

                input_shape = self.input_shape.model_copy(update={'center': Vector.of(
                    (0,) * dna.n_dimension,
                    dimension_size,
                )})

                output_shape = self.output_shape.model_copy(update={'center': Vector.of(
                    (last_idx,) * lower_half_dimension + (0,) * upper_half_dimension,
                    dimension_size,
                )})

                reward_shape = self.reward_shape.model_copy(update={'center': Vector.of(
                    (0,) * lower_half_dimension + (last_idx,) * upper_half_dimension,
                    dimension_size,
                )})

                punish_shape = self.punish_shape.model_copy(update={'center': Vector.of(
                    (last_idx,) * dna.n_dimension,
                    dimension_size,
                )})

                mind_interface = MindInterface(
                    mind=mind,
                    input_shape=input_shape,
                    output_shape=output_shape,
                    reward_shape=reward_shape,
                    punish_shape=punish_shape,
                )

                # measure the fitness
                self.coach.mind = mind_interface
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

            print((num_generations, [f for _, f in self.samples]))
            num_generations += 1


class Activity(Coach):
    """
    Selects for minds with synaptic activity.
    """

    # time of the next input
    n_time: int = 0

    # time when done
    d_time: int = 32

    target_activations_distance: int = 0

    # used to incentivise variation in the activations otherwise we get simple oscillation
    activation_sets: set[tuple[int, ...]] = Field(default_factory=set)

    def input(self, c_time: int):
        if c_time >= self.n_time:
            self.mind.input(c_time, 15)
            self.n_time = c_time + 8

    def output(self, c_time: int):
        # target partial activity to avoid too much and too little activity
        part_neurons = div(len(self.mind.mind.neurons.items), 3)
        self.target_activations_distance += abs(part_neurons - self.mind.mind.num_activations)
        self.activation_sets.add(tuple(self.mind.mind.activation_set))

        if c_time >= self.d_time:
            self.done = True

    def measure(self) -> Fitness:
        return self.target_activations_distance * div(self.d_time ** 2, len(self.activation_sets))

    def reset(self):
        super().reset()
        self.n_time = 0
        self.target_activations_distance = 0
        self.activation_sets = set()


class Echo(Coach):
    # expect these values to be echoed back
    # dict[time, value]
    expectations: dict[int, int] = Field(default_factory=dict)

    # time of the next input
    n_time: int = 0

    # time when to switch from training to measuring
    s_time: int = 250

    # time when done
    d_time: int = 500

    # expect values to be echoed back in this many steps
    delay: int = 4

    fitness: Fitness = 0

    def input(self, c_time: int):
        if c_time >= self.n_time:
            value = rand_int(1, 15)
            self.expectations[c_time + self.delay] = value
            self.mind.input(c_time, value)
            self.n_time += rand_int(1, 10)

    def output(self, c_time: int):
        output = self.mind.output()

        if c_time in self.expectations:
            if output == self.expectations[c_time]:
                if c_time < self.s_time:
                    self.mind.reward(c_time, 15)
            else:
                if c_time < self.s_time:
                    self.mind.punish(c_time, abs(output - self.expectations[c_time]))
                else:
                    self.fitness += abs(output - self.expectations[c_time])

            del self.expectations[c_time]
        elif output == 0:
            if c_time < self.s_time:
                self.mind.reward(c_time, 3)
        else:
            if c_time < self.s_time:
                self.mind.punish(c_time, output)
            else:
                self.fitness += output

        if c_time >= self.d_time:
            self.done = True

    def measure(self) -> Fitness:
        return self.fitness

    def reset(self):
        super().reset()
        self.expectations = {}
        self.n_time = 0
        self.fitness = 0
