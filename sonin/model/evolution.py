import heapq
import time
from datetime import timedelta

from pydantic import BaseModel, Field

from sonin.model.dna import Dna
from sonin.model.hypercube import CityShape, Vector
from sonin.model.mind import Mind, MindInterface
from sonin.model.mutation import DnaMutagen
from sonin.model.step import HasStep
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

    def evolve(
        self,
        initial_sample: DnaMutagen,
        min_generations: int = 1,
        min_elapsed_time: timedelta = timedelta(seconds=0),
    ):
        self.samples = []
        num_generations = 0
        start_time = time.time()

        while num_generations < min_generations or (time.time() - start_time) < min_elapsed_time.total_seconds():
            num_generations += 1
            descendants = []

            for sample, _ in self.samples or [(initial_sample, None)]:
                for _ in range(self.num_descendants):
                    # mutate the DNA
                    descendant = sample.model_copy(deep=True)
                    descendant.mutate(self.num_mutations)

                    # build the descendant mind
                    dna: Dna = descendant.value
                    mind: Mind = dna.build_mind()

                    # configure the interface
                    dimension_size = dna.dimension_size

                    mind_interface = MindInterface(
                        mind=mind,
                        input_shape=CityShape(
                            center=Vector.of((0, 0), dimension_size),
                            size=2,
                        ),
                        output_shape=CityShape(
                            center=Vector.of((dimension_size - 1, 0), dimension_size),
                            size=2,
                        ),
                        reward_shape=CityShape(
                            center=Vector.of((0, dimension_size - 1), dimension_size),
                            size=2,
                        ),
                        punish_shape=CityShape(
                            center=Vector.of((dimension_size - 1, dimension_size - 1), dimension_size),
                            size=2,
                        ),
                    )

                    # measure the fitness
                    self.coach.mind = mind_interface
                    self.coach.reset()
                    c_time = 0

                    while not self.coach.done:
                        self.coach.step(c_time)
                        c_time += 1

                    descendants.append((descendant, self.coach.measure()))

            # keep the most fit
            self.samples = heapq.nsmallest(
                self.sample_retention,
                self.samples + descendants,
                key=lambda x: x[1],
            )

            print((num_generations, [f for _, f in self.samples]))


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
