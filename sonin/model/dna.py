import copy
from typing import Any, Self

from pydantic import BaseModel, Field, field_serializer, field_validator

from sonin.model.fate import Fate, FateTree
from sonin.model.growth import Environment, Incubator
from sonin.model.hypercube import CubeShape, Hypercube, parse_shape, Shape, Vector
from sonin.model.mind import Mind, MindInterface
from sonin.model.neuron import Axon, Neuron
from sonin.model.signal import AffinityDict, Signal, SignalCount, SignalProfile
from sonin.sonin_math import div
from sonin.sonin_random import Pcg32, Random

# list[tuple[Numerator, DenominatorDelta]]
#
# Each item encodes the relative position within the dimension associated with its position in the list. The list must
# be of size num_dimensions. Each position component is calculated as:
#
# dimension_size * Numerator // (Numerator + DenominatorDelta)
#
# (0, n) where n > 0 means: 0 // (0 + n) or 0%
# (n, 0) where n > 0 means: n // (n + 0) or 100%
# (n, n) where n > 0 means: n // (n + n) or 50%
# (3, 1)             means: 3 // (3 + 1) or 75%
# (2, 3)             means: 2 // (2 + 3) or 40%
# [(3, 1), (2, 3)] means: 75% along dimension 0 and 40% along dimension 1
#
# This allows position encodings to remain consistent across changes in dimension size.
type Position = list[tuple[int, int]]


class Dna(BaseModel):
    num_dimensions: int
    dimension_size: int
    max_synapses: int
    max_synapse_strength: int
    max_axon_range: int
    refactory_period: int

    # registry of the signals that exist
    signals: set[Signal] = Field(default_factory=set)

    encoded_environment: list[tuple[Signal, Position, SignalCount]]
    incubation_signals: dict[Signal, SignalCount]
    affinities: AffinityDict
    fate_tree: FateTree
    input_shape: Shape = CubeShape(size=2)
    output_shape: Shape = CubeShape(size=2)
    reward_shape: Shape | None = None
    punish_shape: Shape | None = None

    @field_validator('input_shape', 'output_shape', 'reward_shape', 'punish_shape', mode='before')
    @classmethod
    def validate_shape(cls, shape: dict[str, Any] | None) -> Any | None:
        return shape and parse_shape(shape)

    @field_serializer('input_shape', 'output_shape', 'reward_shape', 'punish_shape')
    @classmethod
    def serialize_shape(cls, shape: BaseModel | None) -> str | dict[str, Any] | None:
        # necessary to call the specific serializer
        return shape and shape.model_dump()

    @classmethod
    def from_defaults(cls) -> Self:
        return Dna(
            num_dimensions=2,
            dimension_size=10,
            max_synapses=1,
            max_synapse_strength=1,
            max_axon_range=1,
            refactory_period=0,
            signals=set(),
            encoded_environment=[],
            incubation_signals={},
            affinities={},
            fate_tree=FateTree(root=Fate.from_defaults()),
        )

    @property
    def environment(self) -> Environment:
        return [
            (
                signal,
                Vector.of(
                    tuple(
                        div(self.dimension_size * numerator, numerator + delta)
                        for numerator, delta in position
                    ),
                ),
                signal_count,
            )
            for signal, position, signal_count in self.encoded_environment
        ]

    @property
    def signal_profile(self) -> SignalProfile:
        return SignalProfile(affinities=copy.deepcopy(self.affinities))

    def build_mind(self, random: Random | None = None) -> MindInterface:
        # perform cell division
        incubator = Incubator(
            num_dimensions=self.num_dimensions,
            dimension_size=self.dimension_size,
            environment=self.environment,
            signal_profile=self.signal_profile,
        )

        incubator.initialize(self.incubation_signals)
        incubator.incubate()

        # determine cell fates
        neuron_items = []

        for cell in incubator.cells:
            fate = self.fate_tree.get_fate(cell.signals)

            neuron_items.append(Neuron(
                position=cell.position,
                axon=Axon(
                    position=cell.position,
                    signals=fate.axon_signals,
                ),
                signals=cell.signals,
                excites=fate.excites,
                activation_level=fate.activation_level,
                refactory_period=fate.refactory_period,
                tetanic_period=fate.tetanic_period,
                stimulation=fate.stimulation,
                overstimulation_threshold=fate.overstimulation_threshold,
            ))

        # construct the mind
        neurons = Hypercube(
            num_dimensions=self.num_dimensions,
            dimension_size=self.dimension_size,
            items=neuron_items,
        )

        mind = Mind(
            max_synapses=self.max_synapses,
            num_dimensions=self.num_dimensions,
            dimension_size=self.dimension_size,
            max_synapse_strength=self.max_synapse_strength,
            axon_range=self.max_axon_range,
            neurons=neurons,
            signal_profile=self.signal_profile,
        )

        mind.random = random or Random(rng=Pcg32())

        # set up the interface
        last_idx = self.dimension_size - 1
        lower_half_dimension = div(self.num_dimensions, 2)
        upper_half_dimension = self.num_dimensions - lower_half_dimension

        input_shape = self.input_shape.model_copy(update={'center': Vector.of(
            (0,) * self.num_dimensions,
            self.dimension_size,
        )})

        output_shape = self.output_shape.model_copy(update={'center': Vector.of(
            (last_idx,) * lower_half_dimension + (0,) * upper_half_dimension,
            self.dimension_size,
        )})

        if self.reward_shape:
            reward_shape = self.reward_shape.model_copy(update={'center': Vector.of(
                (0,) * lower_half_dimension + (last_idx,) * upper_half_dimension,
                self.dimension_size,
            )})
        else:
            reward_shape = None

        if self.punish_shape:
            punish_shape = self.punish_shape.model_copy(update={'center': Vector.of(
                (last_idx,) * self.num_dimensions,
                self.dimension_size,
            )})
        else:
            punish_shape = None

        mind_interface = MindInterface(
            mind=mind,
            input_shape=input_shape,
            output_shape=output_shape,
            reward_shape=reward_shape,
            punish_shape=punish_shape,
        )

        mind.guide_axons()
        mind.randomize_synapses()
        return mind_interface
