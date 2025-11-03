from pydantic import BaseModel

from sonin.model.fate import FateTree
from sonin.model.growth import Incubator
from sonin.model.hypercube import CubeShape, Hypercube, Shape, Vector
from sonin.model.mind import Mind, MindInterface
from sonin.model.neuron import Axon, Neuron
from sonin.model.signal import Signal, SignalCount, SignalProfile
from sonin.sonin_math import div
from sonin.sonin_random import Pcg32, Random


class Dna(BaseModel):
    n_dimension: int
    dimension_size: int
    n_synapse: int
    activation_level: int
    max_neuron_strength: int
    axon_range: int
    refactory_period: int
    environment: list[tuple[Signal, SignalCount, Vector]]
    incubation_signals: dict[Signal, SignalCount]
    signal_profile: SignalProfile
    overstimulation_threshold: int
    fate_tree: FateTree
    input_shape: Shape = CubeShape(size=2)
    output_shape: Shape = CubeShape(size=2)
    reward_shape: Shape | None = None
    punish_shape: Shape | None = None

    def build_mind(self, random: Random | None = None) -> MindInterface:
        # perform cell division
        incubator = Incubator(
            n_dimension=self.n_dimension,
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
                    n_dimension=self.n_dimension,
                    dimension_size=self.dimension_size,
                ),
                signals=cell.signals,
                excites=fate.excites,
                activation_level=fate.activation_level,
                refactory_period=fate.refactory_period,
                tetanic_period=fate.tetanic_period,
                stimulation=fate.stimulation,
            ))

        # construct the mind
        neurons = Hypercube(
            n_dimension=self.n_dimension,
            dimension_size=self.dimension_size,
            items=neuron_items,
        )

        mind = Mind(
            n_synapse=self.n_synapse,
            n_dimension=self.n_dimension,
            dimension_size=self.dimension_size,
            max_neuron_strength=self.max_neuron_strength,
            axon_range=self.axon_range,
            neurons=neurons,
            signal_profile=self.signal_profile,
            overstimulation_threshold=self.overstimulation_threshold,
            random=random or Random(rng=Pcg32()),
        )

        last_idx = self.dimension_size - 1
        lower_half_dimension = div(self.n_dimension, 2)
        upper_half_dimension = self.n_dimension - lower_half_dimension

        input_shape = self.input_shape.model_copy(update={'center': Vector.of(
            (0,) * self.n_dimension,
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
                (last_idx,) * self.n_dimension,
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
