from pytest import fixture

from sonin.model.dna import Dna
from sonin.model.hypercube import Vector
from sonin.model.mind import strengthen_connection, weaken_connection
from sonin.model.neuron import Axon, Neuron
from sonin.model.stimulation import Stimulation


@fixture
def num_dimensions() -> int:
    return 1


@fixture
def dna(num_dimensions: int) -> Dna:
    return Dna.from_defaults(num_dimensions)


@fixture
def neuron_1(dna: Dna) -> Neuron:
    position = Vector.of((0,), dna.dimension_size)

    return Neuron(
        position=position,
        axon=Axon(position=position),
        activation_level=1,
        refactory_period=0,
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
        refactory_period=0,
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
