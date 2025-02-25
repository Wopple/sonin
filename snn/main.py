from random import seed, randint

from model.neuron import ACCEPTING, ACTIVATED, REFACTORY
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
        self.neurons: Hypercube[Neuron] = Hypercube(dna)
        self.neurons.initialize(lambda position: Neuron(dna, position))

    def randomize_synapses(self):
        for n in self.neurons:
            for i in range(self.dna.n_synapse):
                n.synapses[i] = random_position(self.dna)

    def randomize_potential(self):
        for n in self.neurons:
            if randint(0, 1):
                n.potential = self.dna.activation_level
            else:
                n.potential = 0

    def stage_1(self, c_time: int):
        for n in self.neurons:
            if n.state == ACCEPTING and n.potential >= self.dna.activation_level:
                n.activate()
            elif n.state == REFACTORY and n.t_refactory_end <= c_time:
                n.enable()

    def stage_2(self, c_time: int):
        for n in self.neurons:
            if n.state == ACTIVATED:
                n.refactor(c_time)

                for position in n.synapses:
                    post_n = self.neurons.get(position)

                    if post_n.state == ACCEPTING:
                        post_n.potential += 1


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
mind.randomize_potential()

for n in mind.neurons:
    print(n.position.value, [s.value for s in n.synapses])

input_neurons = [n for n in mind.neurons if n.position.value[0] == 0]
output_neurons = [n for n in mind.neurons if n.position.value[0] == 6]

def print_neurons(msg: str, neurons: list[Neuron]):
    print(f"{msg}: {[n.potential for n in neurons]}")

def print_mind():
    ns = []
    for idx, n in enumerate(mind.neurons.items):
        ns.append(n.potential)

        if len(ns) == 7:
            print(ns)
            ns = []

# print_neurons("input", input_neurons)
# print_neurons("output", output_neurons)

print_mind()

for i in range(10):
    mind.stage_1(i)
    mind.stage_2(i)
    print()
    print_mind()
    # print_neurons("input", input_neurons)
    # print_neurons("output", output_neurons)
