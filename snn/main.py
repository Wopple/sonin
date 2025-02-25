from random import seed, randint

from snn.model.dna import Dna
from snn.model.hypercube import Hypercube
from snn.model.neuron import Neuron
from snn.model.position import Position

# Regulate excitation when overstimulated by weakening connections


def random_position(dna: Dna) -> Position:
    return Position(dna.dimension_size, tuple(randint(0, dna.dimension_size - 1) for _ in range(dna.n_dimension)))


class Mind:
    def __init__(self, dna: Dna):
        self.dna: Dna = dna
        self.neurons: Hypercube = Hypercube(dna)
        self.neurons.initialize(lambda position: Neuron(dna, position))

    def randomize_synapses(self):
        for n in self.neurons:
            for i in range(self.dna.n_synapse):
                n.synapses[i] = random_position(self.dna)


seed(0)

dna = Dna(
    min_neurons=100,
    n_synapse=4,
    n_dimension=2,
    activation_level=2,
    refactory_period=2,
)

mind = Mind(dna)
mind.randomize_synapses()

for n in mind.neurons:
    print(n.position.value, [s.value for s in n.synapses])
