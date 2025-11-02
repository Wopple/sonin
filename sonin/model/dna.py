from pydantic import BaseModel

from sonin.model.fate import FateTree
from sonin.model.growth import Incubator
from sonin.model.hypercube import Hypercube, Vector
from sonin.model.mind import Mind
from sonin.model.neuron import Axon, Neuron
from sonin.model.signal import Signal, SignalCount, SignalProfile
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

    def build_mind(self) -> Mind:
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

            # for reproducible initial state
            random=Random(rng=Pcg32()),
        )

        mind.guide_axons()
        mind.randomize_synapses()
        return mind
