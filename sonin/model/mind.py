from random import randint, choice

from sonin.model.dna import Dna
from sonin.model.hypercube import Hypercube, Position
from sonin.model.neuron import ACCEPTING, Neuron
from sonin.model.synapse import Synapse


def random_position(dna: Dna) -> Position:
    return Position(dna.dimension_size, tuple(randint(0, dna.dimension_size - 1) for _ in range(dna.n_dimension)))


def strengthen_connection(pre_neuron: Neuron, post_neuron: Neuron, strength: int, max_strength: int):
    pre_position: Position = pre_neuron.position
    pre_index: int = pre_position.index
    post_position: Position = post_neuron.position
    post_index: int = post_position.index
    synapse: Synapse

    if post_index in pre_neuron.post_synapses:
        synapse = pre_neuron.post_synapses[post_index]
        synapse.strengthen(strength)
    else:
        synapse = Synapse(
            pre_neuron=pre_position,
            post_neuron=post_position,
            strength=strength,
            max_strength=max_strength,
        )

        pre_neuron.post_synapses[post_index] = synapse
        post_neuron.pre_synapses[pre_index] = synapse


def weaken_connection(pre_neuron: Neuron, post_neuron: Neuron, strength: int):
    pre_position: Position = pre_neuron.position
    pre_index: int = pre_position.index
    post_position: Position = post_neuron.position
    post_index: int = post_position.index
    synapse: Synapse

    if post_index in pre_neuron.post_synapses:
        synapse = pre_neuron.post_synapses[post_index]

        if synapse.strength > strength:
            synapse.strength -= strength
        else:
            del pre_neuron.post_synapses[post_index]
            del post_neuron.pre_synapses[pre_index]


class Mind:
    def __init__(self, dna: Dna):
        self.dna: Dna = dna
        self.neurons: Hypercube[Neuron] = Hypercube(dna)
        self.neurons.initialize(lambda position: Neuron(dna, position))

    def randomize_synapses(self):
        for pre_n in self.neurons:
            for i in range(self.dna.n_synapse):
                post_n = self.neurons.get(random_position(self.dna))

                strengthen_connection(
                    pre_neuron=pre_n,
                    post_neuron=post_n,
                    strength=self.dna.max_neuron_strength // 2,
                    max_strength=self.dna.max_neuron_strength,
                )

    def randomize_potential(self):
        for n in self.neurons:
            if randint(0, 1):
                n.potential = self.dna.activation_level
            else:
                n.potential = 0

    def step(self, c_time: int):
        # Iterate multiple times to prevent iteration significance

        for n in self.neurons:
            n.step(c_time)

            if n.stimulation.value > 100 and len(n.pre_synapses) > 0:
                n.stimulation.value = 0
                syn: Synapse = choice(list(n.pre_synapses.values()))

                weaken_connection(
                    pre_neuron=self.neurons.get(syn.pre_neuron),
                    post_neuron=n,
                    strength=self.dna.max_neuron_strength // 2,
                )

        for n in self.neurons:
            if n.activated:
                self.propagate_potential(n)
                self.strengthen_simultaneous_activation(n)

        for n in self.neurons:
            if n.activated:
                n.deactivate()

    def propagate_potential(self, pre_n: Neuron):
        for syn in pre_n.post_synapses.values():
            post_n = self.neurons.get(syn.post_neuron)

            if post_n.state == ACCEPTING:
                if pre_n.excites:
                    post_n.potential += syn.strength
                else:
                    post_n.potential -= syn.strength

    def strengthen_simultaneous_activation(self, pre_n: Neuron):
        for syn in pre_n.post_synapses.values():
            post_n = self.neurons.get(syn.post_neuron)

            if pre_n.position.index != post_n.position.index and post_n.activated:
                syn.strengthen(1)
