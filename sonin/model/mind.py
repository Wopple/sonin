from dataclasses import dataclass, field
from random import choice, randint

from sonin.model.hypercube import Hypercube, Vector
from sonin.model.neuron import ACCEPTING, Axon, Neuron
from sonin.model.signal import SignalProfile
from sonin.model.synapse import Synapse
from sonin.sonin_math import div


def strengthen_connection(pre_neuron: Neuron, post_neuron: Neuron, strength: int, max_strength: int):
    pre_position: Vector = pre_neuron.position
    pre_index: int = pre_position.index
    post_position: Vector = post_neuron.position
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
    pre_position: Vector = pre_neuron.position
    pre_index: int = pre_position.index
    post_position: Vector = post_neuron.position
    post_index: int = post_position.index
    synapse: Synapse

    if post_index in pre_neuron.post_synapses:
        synapse = pre_neuron.post_synapses[post_index]

        if synapse.strength > strength:
            synapse.strength -= strength
        else:
            # disconnect the synapse when the strength reaches zero
            del pre_neuron.post_synapses[post_index]
            del post_neuron.pre_synapses[pre_index]


@dataclass
class Mind:
    n_synapse: int
    n_dimension: int
    dimension_size: int
    max_neuron_strength: int
    axon_range: int
    neurons: Hypercube[Neuron] = field(init=False)
    signal_profile: SignalProfile = field(default_factory=SignalProfile)

    def __post_init__(self):
        self.neurons = Hypercube(
            n_dimension=self.n_dimension,
            dimension_size=self.dimension_size,
        )

    def initialize(self, activation_level: int, refactory_period: int):
        self.neurons.initialize(lambda position: Neuron(
            position=position,
            axon=Axon(position, self.n_dimension, self.dimension_size),
            activation_level=activation_level,
            refactory_period=refactory_period,
        ))

        self.guide_axons()

    def random_position(self, center: Vector, distance: int) -> Vector:
        """
        Returns a vector within `distance` city blocks from the center wrapping at the dimension size.
        """
        def random_component(idx: int) -> int:
            return (center.value[idx] + randint(-distance, distance)) % self.dimension_size

        return Vector(self.dimension_size, tuple(random_component(idx) for idx in range(self.n_dimension)))

    def randomize_synapses(self):
        for pre_n in self.neurons:
            for i in range(self.n_synapse):
                post_n = self.neurons.get(self.random_position(pre_n.axon.position, self.axon_range))

                strengthen_connection(
                    pre_neuron=pre_n,
                    post_neuron=post_n,
                    strength=div(self.max_neuron_strength, 2),
                    max_strength=self.max_neuron_strength,
                )

    def randomize_potential(self):
        for n in self.neurons:
            if randint(0, 1):
                n.potential = n.activation_level
            else:
                n.potential = 0

    def guide_axons(self):
        all_signals = [
            (signal, n.position, n.effective_range.get(signal, self.n_dimension * self.dimension_size))
            for n in self.neurons for signal in n.signals
        ]

        for n in self.neurons:
            axon = n.axon
            axon_position = axon.position
            growth_signals = axon.signals
            past_positions = set()
            direction = axon.direction

            # Stop if trying to move to a past position
            while axon_position not in past_positions:
                past_positions.add(axon_position)
                attraction = Vector(self.dimension_size, tuple(0 for _ in range(self.n_dimension)))

                # Sum the attractive effects between the signals
                for guide_signal, location, effective_range in all_signals:
                    distance = location.city_distance(axon_position)

                    # do not apply signals out of range
                    if distance <= effective_range:
                        for growth_signal in growth_signals:
                            degree_of_attraction = self.signal_profile.attraction(growth_signal, guide_signal)

                            # dividing by distance to weaken farther signals
                            attraction += div((location - axon_position) * degree_of_attraction, distance)

                # Stop if the net attraction is zero
                if all(c == 0 for c in attraction.value):
                    break

                # Let the previous direction affect the new direction
                direction = (direction + attraction).city_unit()

                # Move the axon clipping to the hypercube size
                axon_position = (axon_position + direction).clip()

            axon.position = axon_position

    def step(self, c_time: int):
        """ Advance the mind forward one step. `c_time` is a monotonically increasing step number. """

        # Iterate multiple times so always writing to data not being read from.
        # This makes the algorithm trivial to parallelize.

        for n in self.neurons:
            n.step(c_time)

            if n.stimulation.value > 100 and len(n.pre_synapses) > 0:
                n.stimulation.value = 0
                syn: Synapse = choice(list(n.pre_synapses.values()))

                weaken_connection(
                    pre_neuron=self.neurons.get(syn.pre_neuron),
                    post_neuron=n,
                    strength=div(self.max_neuron_strength, 2),
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
