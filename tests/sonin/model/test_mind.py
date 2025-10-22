from pytest import fixture

from sonin.model.dna import Dna
from sonin.model.hypercube import Vector
from sonin.model.mind import strengthen_connection, weaken_connection
from sonin.model.neuron import Axon, Neuron


@fixture
def n_dimension() -> int:
    return 1


@fixture
def dna(n_dimension: int) -> Dna:
    return Dna(min_neurons=2, n_dimension=n_dimension)


@fixture
def neuron_1(dna: Dna) -> Neuron:
    position = Vector(value=(0,), dimension_size=dna.n_dimension)

    return Neuron(
        position=position,
        axon=Axon(position, dna.n_dimension, dna.dimension_size),
        activation_level=dna.activation_level,
        refactory_period=dna.refactory_period,
    )


@fixture
def neuron_2(dna: Dna) -> Neuron:
    position = Vector(value=(1,), dimension_size=dna.n_dimension)

    return Neuron(
        position=position,
        axon=Axon(position, dna.n_dimension, dna.dimension_size),
        activation_level=dna.activation_level,
        refactory_period=dna.refactory_period,
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
