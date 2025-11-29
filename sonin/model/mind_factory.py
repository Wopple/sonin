from sonin.model.dna import Dna
from sonin.model.fate import Fate
from sonin.model.hypercube import AbsPosition, Hypercube, Vector
from sonin.model.metric import Metric
from sonin.model.mind import Mind, MindInterface
from sonin.model.neuron import Axon, Neuron
from sonin.sonin_math import div
from sonin.sonin_random import Random


class MindFactory:
    def __init__(self, dna: Dna):
        self.dna = dna
        self.paint_count_metric = Metric()

    def build_mind(self, random: Random) -> MindInterface:
        # determine cell fates
        fate_positions: list[tuple[Vector, Fate] | None] = [None] * self.dna.num_neurons
        fate_paints = self.dna.fate_paints.copy()
        fate_paints.reverse()

        for paint, fate in fate_paints:
            num_paints = 0

            for position in paint.positions(self.dna.num_dimensions, self.dna.dimension_size):
                if fate_positions[position.index] is None:
                    fate_positions[position.index] = position, fate
                    num_paints += 1

            self.paint_count_metric.record(num_paints)

        # create the neurons
        neuron_items = [
            Neuron(
                position=position,
                # TODO: consider preventing axons from centering on input neurons since they cannot form synapses
                axon=Axon(position=(position + Vector.of(fate.axon_offset, self.dna.dimension_size)).clip()),
                excites=fate.excites,
                activation_level=fate.activation_level,
                refactory_period=fate.refactory_period,
                tetanic_period=fate.tetanic_period,
                stimulation=fate.stimulation,
                overstimulation_threshold=fate.overstimulation_threshold,
            )
            for position, fate in fate_positions
        ]

        # construct the mind
        neurons = Hypercube(
            num_dimensions=self.dna.num_dimensions,
            dimension_size=self.dna.dimension_size,
            items=neuron_items,
        )

        mind = Mind(
            num_dimensions=self.dna.num_dimensions,
            dimension_size=self.dna.dimension_size,
            max_synapses=self.dna.max_synapses,
            max_synapse_strength=self.dna.max_synapse_strength,
            max_axon_range=self.dna.max_axon_range,
            neurons=neurons,
        )

        mind.random = random

        # set up the interface
        last_idx = self.dna.dimension_size - 1
        lower_half_dimension = div(self.dna.num_dimensions, 2)
        upper_half_dimension = self.dna.num_dimensions - lower_half_dimension

        # corner with indices of all 0
        input_center = Vector.of(
            (0,) * self.dna.num_dimensions,
            self.dna.dimension_size,
            )

        input_shape = self.dna.input_shape.model_copy(update={'center': AbsPosition(value=input_center)})

        # corner opposite input
        output_center = Vector.of(
            (last_idx,) * lower_half_dimension + (0,) * upper_half_dimension,
            self.dna.dimension_size,
            )

        output_shape = self.dna.output_shape.model_copy(update={'center': AbsPosition(value=output_center)})

        if self.dna.reward_shape:
            # corner with half indices of 0 and half indices of the maximum
            reward_center = Vector.of(
                (0,) * lower_half_dimension + (last_idx,) * upper_half_dimension,
                self.dna.dimension_size,
                )

            reward_shape = self.dna.reward_shape.model_copy(update={'center': AbsPosition(value=reward_center)})
        else:
            reward_shape = None

        if self.dna.punish_shape:
            # corner opposite reward
            punish_center = Vector.of(
                (last_idx,) * self.dna.num_dimensions,
                self.dna.dimension_size,
                )

            punish_shape = self.dna.punish_shape.model_copy(update={'center': AbsPosition(value=punish_center)})
        else:
            punish_shape = None

        mind_interface = MindInterface(
            mind=mind,
            input_shape=input_shape,
            output_shape=output_shape,
            reward_shape=reward_shape,
            punish_shape=punish_shape,
        )

        mind.randomize_synapses()
        return mind_interface
