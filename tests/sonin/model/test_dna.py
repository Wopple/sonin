from pytest import mark

from sonin.model.dna import Dna


@mark.parametrize(
    'min_neurons, n_dimension, expected',
    [
        (1, 1, 1),
        (2, 1, 2),
        (3, 1, 3),
        (1, 2, 1),
        (2, 2, 2),
        (3, 2, 2),
        (4, 2, 2),
        (5, 2, 3),
        (1, 3, 1),
        (2, 3, 2),
    ]
)
def test_dimension_size(min_neurons: int, n_dimension: int, expected: int):
    dna = Dna(
        min_neurons=min_neurons,
        n_dimension=n_dimension,
    )

    assert dna.dimension_size == expected


@mark.parametrize(
    'min_neurons, n_dimension, expected',
    [
        (1, 1, 1),
        (2, 1, 2),
        (3, 1, 3),
        (1, 2, 1),
        (2, 2, 4),
        (3, 2, 4),
        (4, 2, 4),
        (5, 2, 9),
        (1, 3, 1),
        (2, 3, 8),
    ]
)
def test_n_neuron(min_neurons: int, n_dimension: int, expected: int):
    dna = Dna(
        min_neurons=min_neurons,
        n_dimension=n_dimension,
    )

    assert dna.n_neuron == expected
