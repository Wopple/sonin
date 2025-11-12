from pytest import fixture

from sonin.model.dna import Dna
from sonin.model.fate import FateTree
from sonin.model.hypercube import Hypercube, Vector
from sonin.model.mind import Mind, strengthen_connection, weaken_connection
from sonin.model.neuron import Axon, Neuron
from sonin.model.signal import Signal, SignalCount, SignalProfile
from sonin.model.stimulation import Stimulation


@fixture
def num_dimensions() -> int:
    return 1


@fixture
def dna(num_dimensions: int) -> Dna:
    return Dna(
        num_dimensions=num_dimensions,
        dimension_size=2,
        max_synapses=1,
        max_synapse_strength=1,
        max_axon_range=1,
        refactory_period=1,
        encoded_environment=[],
        incubation_signals={},
        affinities={},
        fate_tree=FateTree(),
    )


@fixture
def neuron_1(dna: Dna) -> Neuron:
    position = Vector.of((0,), dna.dimension_size)

    return Neuron(
        position=position,
        axon=Axon(position=position),
        activation_level=1,
        refactory_period=dna.refactory_period,
        stimulation=Stimulation(),
        overstimulation_threshold=1,
    )


@fixture
def neuron_2(dna: Dna) -> Neuron:
    position = Vector.of((1,), dna.dimension_size)

    return Neuron(
        position=position,
        axon=Axon(position=position),
        activation_level=1,
        refactory_period=dna.refactory_period,
        stimulation=Stimulation(),
        overstimulation_threshold=1,
    )


def test_strengthen_connection_new(neuron_1: Neuron, neuron_2: Neuron):
    assert neuron_2.position.index not in neuron_1.post_synapses
    assert neuron_1.position.index not in neuron_2.pre_synapses

    strengthen_connection(neuron_1, neuron_2, 1, 4)

    assert neuron_2.position.index in neuron_1.post_synapses
    assert neuron_1.position.index in neuron_2.pre_synapses
    assert neuron_1.post_synapses[neuron_2.position.index] is neuron_2.pre_synapses[neuron_1.position.index]
    assert neuron_1.post_synapses[neuron_2.position.index].strength == 1


def test_strengthen_connection_existing(neuron_1: Neuron, neuron_2: Neuron):
    strengthen_connection(neuron_1, neuron_2, 1, 4)
    strengthen_connection(neuron_1, neuron_2, 1, 4)

    assert neuron_2.position.index in neuron_1.post_synapses
    assert neuron_1.position.index in neuron_2.pre_synapses
    assert neuron_1.post_synapses[neuron_2.position.index] is neuron_2.pre_synapses[neuron_1.position.index]
    assert neuron_1.post_synapses[neuron_2.position.index].strength == 2


def test_weaken_connection_retain(neuron_1: Neuron, neuron_2: Neuron):
    strengthen_connection(neuron_1, neuron_2, 1, 4)
    strengthen_connection(neuron_1, neuron_2, 1, 4)
    weaken_connection(neuron_1, neuron_2, 1)

    assert neuron_2.position.index in neuron_1.post_synapses
    assert neuron_1.position.index in neuron_2.pre_synapses
    assert neuron_1.post_synapses[neuron_2.position.index] is neuron_2.pre_synapses[neuron_1.position.index]
    assert neuron_1.post_synapses[neuron_2.position.index].strength == 1


def test_weaken_connection_eliminate(neuron_1: Neuron, neuron_2: Neuron):
    strengthen_connection(neuron_1, neuron_2, 1, 4)
    weaken_connection(neuron_1, neuron_2, 1)

    assert neuron_2.position.index not in neuron_1.post_synapses
    assert neuron_1.position.index not in neuron_2.pre_synapses


def test_guide_axons():
    num_dimensions = 2
    dimension_size = 5

    signal_lookup = {
        (0, 0): (
            {0: 1},
            {0: 1},
        ),
        (4, 4): (
            {1: 1},
            {1: 1},
        ),
    }

    def create_neuron(position: Vector) -> Neuron:
        neuron_signals, axon_signals = signal_lookup.get(position.value, ({}, {}))

        return Neuron(
            position=position,
            axon=Axon(
                position=position,
                signals=axon_signals,
            ),
            signals=neuron_signals,
        )

    neurons = Hypercube[Neuron](num_dimensions=num_dimensions, dimension_size=dimension_size)
    neurons.initialize(create_neuron)

    mind = Mind(
        max_synapses=1,
        num_dimensions=num_dimensions,
        dimension_size=dimension_size,
        max_synapse_strength=1,
        axon_range=1,
        neurons=neurons,
        signal_profile=SignalProfile(affinities={
            0: {1: 2},
            1: {0: 2},
        }),
    )

    mind.guide_axons()

    assert neurons.get((0, 0)).axon.position.value == (4, 4)
    assert neurons.get((4, 4)).axon.position.value == (0, 0)
