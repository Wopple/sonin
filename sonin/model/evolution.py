import heapq
import time
from datetime import timedelta

from pydantic import BaseModel, Field

from sonin.model.dna import Dna
from sonin.model.growth import Incubator
from sonin.model.hypercube import CityShape, Hypercube, Vector
from sonin.model.mind import Mind, MindInterface
from sonin.model.mutation import DnaMutagen
from sonin.model.neuron import Axon, Neuron
from sonin.model.step import HasStep
from sonin.sonin_random import Pcg32, Random

type Fitness = int


class FitnessCriteria(BaseModel):
    """
    Records metrics from a mind for the purpose of producing a final measurement of fitness.
    """

    mind: MindInterface | None = None

    def record(self, c_time: int):
        pass

    def measure(self) -> Fitness:
        raise NotImplementedError(f"{self.__class__.__name__}.measure")

    def reset(self):
        pass


class Coach(BaseModel, HasStep):
    """
    Interacts with a mind to elicit behavior. Rewards good behavior and punishes bad behavior.
    """

    mind: MindInterface | None = None
    done: bool = False

    def step(self, c_time: int):
        self.mind.step(c_time)

    def reset(self):
        self.done = False


class PetriDish(BaseModel):
    """
    Evolves DNA selecting the best fitness.
    """
    samples: dict[DnaMutagen, Fitness] = Field(min_length=1)

    # performance heuristic
    fitness_criteria: FitnessCriteria

    # exercises the mind
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
        self.samples = {initial_sample: 0}
        num_generations = 0
        start_time = time.time()

        while num_generations < min_generations and (time.time() - start_time) < min_elapsed_time.total_seconds():
            num_generations += 1

            for sample in self.samples.keys():
                for _ in range(self.num_descendants):
                    # mutate the DNA
                    descendant = sample.model_copy(deep=True)
                    descendant.mutate(self.num_mutations)
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

                    # measure fitness
                    self.fitness_criteria.mind = mind_interface
                    self.fitness_criteria.reset()
                    self.coach.mind = mind_interface
                    self.coach.reset()
                    c_time = 0

                    while not self.coach.done:
                        self.coach.step(c_time)
                        self.fitness_criteria.record(c_time)
                        c_time += 1

                    self.samples[descendant] = self.fitness_criteria.measure()

            # keep the most fit
            self.samples = dict(heapq.nlargest(
                self.sample_retention,
                self.samples.items(),
                key=lambda x: x[1],
            ))
