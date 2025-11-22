from typing import Any, Self

from pydantic import BaseModel

from sonin.model.fate import Fate
from sonin.model.hypercube import AbsPosition, Hypercube, Vector
from sonin.model.mind import Mind, MindInterface
from sonin.model.neuron import Axon, Neuron
from sonin.model.paint import CompleteFill, FillShape, RectangleShape, Shape
from sonin.sonin_math import div
from sonin.sonin_random import Random


class Dna(BaseModel):
    num_dimensions: int
    dimension_size: int
    max_synapses: int
    max_synapse_strength: int
    max_axon_range: int

    # the first paint must always be a FillShape-CompleteFill paint to serve as a default
    fate_paints: list[tuple[Shape, Fate]]

    input_shape: Shape = None
    output_shape: Shape = None
    reward_shape: Shape | None = None
    punish_shape: Shape | None = None

    def model_post_init(self, context: Any, /):
        assert isinstance(self.fate_paints[0][0], FillShape)
        assert isinstance(self.fate_paints[0][0].fill, CompleteFill)

        if self.input_shape is None:
            self.input_shape = RectangleShape(sizes=(3,) * self.num_dimensions)

        if self.output_shape is None:
            self.output_shape = RectangleShape(sizes=(3,) * self.num_dimensions)

    @property
    def num_neurons(self) -> int:
        return self.dimension_size ** self.num_dimensions

    @classmethod
    def from_defaults(cls, num_dimensions: int = 2, dimension_size: int = 10) -> Self:
        return Dna(
            num_dimensions=num_dimensions,
            dimension_size=dimension_size,
            max_synapses=1,
            max_synapse_strength=1,
            max_axon_range=1,
            fate_paints=[(FillShape(fill=CompleteFill()), Fate.from_defaults(num_dimensions))],
        )

    def build_mind(self, random: Random) -> MindInterface:
        # determine cell fates
        fate_positions: list[tuple[Vector, Fate] | None] = [None] * self.num_neurons
        fate_paints = self.fate_paints.copy()
        fate_paints.reverse()

        for paint, fate in fate_paints:
            for position in paint.positions(self.num_dimensions, self.dimension_size):
                if fate_positions[position.index] is None:
                    fate_positions[position.index] = position, fate

        # create the neurons
        neuron_items = [
            Neuron(
                position=position,
                # TODO: consider preventing axons from centering on input neurons since they cannot form synapses
                axon=Axon(position=(position + Vector.of(fate.axon_offset, self.dimension_size)).clip()),
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
            num_dimensions=self.num_dimensions,
            dimension_size=self.dimension_size,
            items=neuron_items,
        )

        mind = Mind(
            num_dimensions=self.num_dimensions,
            dimension_size=self.dimension_size,
            max_synapses=self.max_synapses,
            max_synapse_strength=self.max_synapse_strength,
            max_axon_range=self.max_axon_range,
            neurons=neurons,
        )

        mind.random = random

        # set up the interface
        last_idx = self.dimension_size - 1
        lower_half_dimension = div(self.num_dimensions, 2)
        upper_half_dimension = self.num_dimensions - lower_half_dimension

        input_center = Vector.of(
            (0,) * self.num_dimensions,
            self.dimension_size,
        )

        input_shape = self.input_shape.model_copy(update={'center': AbsPosition(value=input_center)})

        output_center = Vector.of(
            (last_idx,) * lower_half_dimension + (0,) * upper_half_dimension,
            self.dimension_size,
        )

        output_shape = self.output_shape.model_copy(update={'center': AbsPosition(value=output_center)})

        if self.reward_shape:
            reward_center = Vector.of(
                (0,) * lower_half_dimension + (last_idx,) * upper_half_dimension,
                self.dimension_size,
            )

            reward_shape = self.reward_shape.model_copy(update={'center': AbsPosition(value=reward_center)})
        else:
            reward_shape = None

        if self.punish_shape:
            punish_center = Vector.of(
                (last_idx,) * self.num_dimensions,
                self.dimension_size,
            )

            punish_shape = self.punish_shape.model_copy(update={'center': AbsPosition(value=punish_center)})
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
