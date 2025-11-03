from typing import Any, Generator

from pydantic import BaseModel, Field

from sonin.model.hypercube import Hypercube, Shape, Vector, VectorIndex
from sonin.model.neuron import ACCEPTING, Neuron
from sonin.model.signal import Signal, SignalCount, SignalProfile
from sonin.model.step import HasStep
from sonin.model.synapse import Synapse
from sonin.sonin_math import div
from sonin.sonin_random import Random


def strengthen_connection(pre_neuron: Neuron, post_neuron: Neuron, strength: int, max_strength: int):
    pre_position: Vector = pre_neuron.position
    pre_index: VectorIndex = pre_position.index
    post_position: Vector = post_neuron.position
    post_index: VectorIndex = post_position.index
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
    pre_index: VectorIndex = pre_position.index
    post_position: Vector = post_neuron.position
    post_index: VectorIndex = post_position.index
    synapse: Synapse

    if post_index in pre_neuron.post_synapses:
        synapse = pre_neuron.post_synapses[post_index]

        if synapse.strength > strength:
            synapse.strength -= strength
        else:
            # disconnect the synapse when the strength reaches zero
            del pre_neuron.post_synapses[post_index]
            del post_neuron.pre_synapses[pre_index]


class Mind(BaseModel, HasStep):
    # mutable properties
    n_synapse: int
    n_dimension: int
    dimension_size: int
    max_neuron_strength: int
    axon_range: int
    neurons: Hypercube[Neuron]
    signal_profile: SignalProfile
    overstimulation_threshold: int

    # interface
    input_indices: set[VectorIndex] = set()

    # util
    random: Random = Field(default_factory=Random)

    # debug
    print_activations: bool = False

    # metrics
    num_activations: int = 0
    activation_set: list[int] = Field(default_factory=list)
    effective_range: dict[Signal, int] = Field(default_factory=dict)

    def positions_in_range(self, position: Vector, exclude_input: bool = False) -> Generator[Vector, None, None]:
        """
        Yield all vectors near the position clipping at the border. The area yielded is a hypercube with radius equal
        to the axon range. Can exclude input neurons to prevent them from being activated by anything other than input.
        """

        # [(negative, positive)]
        clipped_ranges: list[tuple[int, int]] = [
            (max(-self.axon_range, -v), min(self.axon_range, self.dimension_size - 1 - v))
            for v in position.value
        ]

        def iterate(dim: int, value_part: tuple[int, ...] = ()) -> Generator[tuple[int, ...], None, None]:
            if dim == 0:
                yield value_part
            else:
                for v in range(*clipped_ranges[dim - 1]):
                    yield from iterate(dim - 1, (v,) + value_part)

        for value in iterate(self.n_dimension):
            if (not exclude_input) or position.index not in self.input_indices:
                yield Vector.of(value, self.dimension_size)

    def random_position(self, center: Vector, exclude_input: bool = False) -> Vector | None:
        """
        Returns a random position in range.
        """

        candidates = tuple(p for p in self.positions_in_range(center, exclude_input=exclude_input))

        if candidates:
            return self.random.choice(candidates)
        else:
            return None

    def randomize_synapses(self):
        for pre_n in self.neurons:
            for i in range(self.n_synapse):
                post_n = self.neurons.get(self.random_position(pre_n.axon.position, exclude_input=True))

                if post_n:
                    strengthen_connection(
                        pre_neuron=pre_n,
                        post_neuron=post_n,
                        strength=div(self.max_neuron_strength, 2),
                        max_strength=self.max_neuron_strength,
                    )

    def randomize_potential(self):
        for n in self.neurons:
            if self.random.rand_bool():
                n.potential = n.activation_level
            else:
                n.potential = 0

    # TODO: consider preventing axons from centering on input neurons
    def guide_axons(self):
        max_range = self.n_dimension * self.dimension_size

        all_signals: list[tuple[Signal, SignalCount, Vector]] = [
            (signal, signal_count, n.position)
            for n in self.neurons for signal, signal_count in n.signals.items()
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

                attraction = Vector.of((0,) * self.n_dimension, self.dimension_size)

                # Sum the attractive effects between the signals
                for guide_signal, guide_signal_count, guide_position, effective_range in all_signals:
                    distance = guide_position.city_distance(axon_position)

                    # skip signals that are out of range
                    if distance > self.effective_range.get(guide_signal, max_range):
                        continue

                    # skip signals that are behind the axon
                    if direction * (guide_position - axon_position) < 0:
                        continue

                    for growth_signal, growth_signal_count in growth_signals.items():
                        attraction += self.signal_profile.attraction_force(
                            growth_signal,
                            guide_signal,
                            axon_position,
                            guide_position,
                            growth_signal_count * guide_signal_count,
                        )

                # Stop if the net attraction is zero
                if all(c == 0 for c in attraction.value):
                    break

                # Let the previous direction affect the new direction
                direction = (direction + attraction).city_unit()

                # Move the axon clipping to the hypercube size
                axon_position = (axon_position + direction).clip()

            axon.position = axon_position

    def step(self, c_time: int):
        """
        Advance the mind forward one step.
        `c_time` is a monotonically increasing step number.
        """

        self.num_activations = 0
        self.activation_set = []

        # Iterate multiple times to avoid reading from and writing to the same data.
        # This makes the algorithm trivial to parallelize at a later time.

        for idx, n in enumerate(self.neurons):
            previous_activated = n.activated
            n.step(c_time)

            if idx % 64 == 0:
                self.activation_set.append(0)

            if (not previous_activated) and n.activated:
                self.num_activations += 1
                self.activation_set[-1] |= 1 << (idx % 64)

            # prevent overstimulation
            if n.stimulation and n.stimulation.value > self.overstimulation_threshold and len(n.pre_synapses) > 0:
                n.stimulation.value = 0
                pre_syns = list(n.pre_synapses.values())

                # find the input neuron with the most stimulation to maximize the effect
                most_stimulated_pre_n = self.neurons.get(pre_syns[0].pre_neuron)

                for s in pre_syns[1:]:
                    candidate = self.neurons.get(s.pre_neuron)

                    if candidate.stimulation.value > most_stimulated_pre_n.stimulation.value:
                        most_stimulated_pre_n = candidate

                # weaken the connection if it excites
                if most_stimulated_pre_n.excites:
                    weaken_connection(
                        pre_neuron=most_stimulated_pre_n,
                        post_neuron=n,
                        strength=div(self.max_neuron_strength, 2),
                    )
                # strengthen the connection if it depresses
                else:
                    strengthen_connection(
                        pre_neuron=most_stimulated_pre_n,
                        post_neuron=n,
                        strength=div(self.max_neuron_strength, 2),
                        max_strength=self.max_neuron_strength,
                    )

        for n in self.neurons:
            if n.activated:
                self.propagate_potential(n)

            if c_time % 64 == n.position.index % 64:
                self.strengthen_simultaneous_activations(n)

        if self.print_activations:
            s = ''

            for x in range(self.dimension_size):
                for y in range(self.dimension_size):
                    s += '[]' if self.neurons.get((x, y)).activated else '  '

                s += '\n'

            print((self.num_activations, self.activation_set))
            print(s)

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

    def strengthen_simultaneous_activations(self, pre_n: Neuron):
        """
        This is essentially Hebbian neuroplasticity: neurons the fire together, wire together.
        """

        def are_correlated(pre: Neuron, post: Neuron) -> bool:
            # pre-activations are shifted right to see if they preceded post-activations
            #
            # c_time 1
            #   pre:  0001
            #   post: 0000
            # c_time 2
            #   pre:  0010
            #   post: 0001
            return bin((pre.recent_activations >> 1) & post.recent_activations).count("1") > 0

        # form a new synapse
        candidates = []

        if len(pre_n.post_synapses) < self.n_synapse:
            for position in self.positions_in_range(pre_n.axon.position, exclude_input=True):
                if position != pre_n.position and position.index not in pre_n.post_synapses:
                    post_n = self.neurons.get(position)

                    if are_correlated(pre_n, post_n):
                        candidates.append(post_n)

        if candidates:
            strengthen_connection(pre_n, self.random.choice(candidates), 1, self.max_neuron_strength)

        # strengthen existing synapses
        for syn in pre_n.post_synapses.values():
            if are_correlated(pre_n, self.neurons.get(syn.post_neuron)):
                syn.strengthen(1)


class MindInterface(BaseModel, HasStep):
    mind: Mind
    input_shape: Shape
    output_shape: Shape
    reward_shape: Shape | None = None
    punish_shape: Shape | None = None
    input_neurons: list[Neuron] = None
    output_neurons: list[Neuron] = None
    reward_neurons: list[Neuron] = None
    punish_neurons: list[Neuron] = None

    def model_post_init(self, context: Any, /):
        input_indices: set[VectorIndex] = set()

        self.input_neurons = [self.mind.neurons.get(p) for p in self.input_shape.positions()]
        input_indices.union({p.index for p in self.input_shape.positions()})

        self.output_neurons = [self.mind.neurons.get(p) for p in self.output_shape.positions()]

        if self.reward_shape:
            self.reward_neurons = [self.mind.neurons.get(p) for p in self.reward_shape.positions()]
            input_indices.union({p.index for p in self.reward_shape.positions()})

        if self.punish_shape:
            self.punish_neurons = [self.mind.neurons.get(p) for p in self.punish_shape.positions()]
            input_indices.union({p.index for p in self.punish_shape.positions()})

        self.mind.input_indices = input_indices

    def step(self, c_time: int):
        self.mind.step(c_time)

    @staticmethod
    def activate_by(c_time: int, value: int, neurons: list[Neuron]):
        """
        Each bit in value is mapped to a neuron to activate. In this way, some neurons are activated only when value
        is high. That can be interpreted as a more potent signal.
        """
        # If the value is larger than the value which activates all neurons, it should also activate all neurons.
        value = min(value, (2 << len(neurons)) - 1)

        for i in range(len(neurons)):
            if value & (1 << i):
                neurons[i].activate(c_time)

    def input(self, c_time: int, value: int):
        MindInterface.activate_by(c_time, value, self.input_neurons)

    def output(self) -> int:
        value = 0

        for i in range(len(self.output_neurons)):
            if self.output_neurons[i].activated:
                value |= 1 << i

        return value

    def reward(self, c_time: int, value: int):
        MindInterface.activate_by(c_time, value, self.reward_neurons)

    def punish(self, c_time: int, value: int):
        MindInterface.activate_by(c_time, value, self.punish_neurons)
