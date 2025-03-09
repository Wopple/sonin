from random import randint

from sonin.model.dna import Dna
from sonin.model.hypercube import Hypercube, Position
from sonin.model.neuron import ACCEPTING, ACTIVATED, Neuron
from sonin.model.synapse import Synapse


def random_position(dna: Dna) -> Position:
    return Position(dna.dimension_size, tuple(randint(0, dna.dimension_size - 1) for _ in range(dna.n_dimension)))


class Mind:
    def __init__(self, dna: Dna):
        self.dna: Dna = dna
        self.neurons: Hypercube[Neuron] = Hypercube(dna)
        self.neurons.initialize(lambda position: Neuron(dna, position))

    def randomize_synapses(self):
        for n in self.neurons:
            for i in range(self.dna.n_synapse):
                n.synapses[i] = Synapse(1, n.position, random_position(self.dna))

    def randomize_potential(self):
        for n in self.neurons:
            if randint(0, 1):
                n.potential = self.dna.activation_level
            else:
                n.potential = 0

    def step(self, c_time: int):
        for n in self.neurons:
            n.step(c_time)

        for pre_n in self.neurons:
            if pre_n.state == ACTIVATED:
                pre_n.refactor(c_time)

                for syn in pre_n.synapses:
                    if syn.post_neuron is not None:
                        post_n = self.neurons.get(syn.post_neuron)

                        if post_n.state == ACCEPTING:
                            if pre_n.excites:
                                post_n.potential += syn.strength
                            else:
                                post_n.potential -= syn.strength
