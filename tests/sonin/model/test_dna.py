from pytest import mark
from random import seed

from sonin.model.dna import Dna, IntMutagen, UintMutagen


@mark.parametrize(
    'seed_value, value, min_value, max_value, deviation_weight, expected',
    [
        # should vary up or down by exactly 1
        (0, 0, -1, 1, 1, 1),
        (1, 0, -1, 1, 1, -1),
        (2, 0, -1, 1, 1, -1),
        (3, 0, -1, 1, 1, -1),
        (4, 0, -1, 1, 1, -1),
        (5, 0, -1, 1, 1, 1),

        # should vary up or down by exactly 1 instead of 1-9 due to min and max limits
        (0, 0, -1, 1, 9, 1),
        (1, 0, -1, 1, 9, -1),
        (2, 0, -1, 1, 9, -1),
        (3, 0, -1, 1, 9, -1),
        (4, 0, -1, 1, 9, -1),
        (5, 0, -1, 1, 9, 1),

        # should vary up or down by exactly 1 instead of by 1-9 because deviation_weight is only 1
        (0, 0, -9, 9, 1, 1),
        (1, 0, -9, 9, 1, -1),
        (2, 0, -9, 9, 1, -1),
        (3, 0, -9, 9, 1, -1),
        (4, 0, -9, 9, 1, -1),
        (5, 0, -9, 9, 1, 1),

        # should vary up or down by 1-3 instead of by 1-9 because deviation_weight is only 3
        (0, 0, -9, 9, 3, 2),
        (1, 0, -9, 9, 3, -3),
        (2, 0, -9, 9, 3, -1),
        (3, 0, -9, 9, 3, -3),
        (4, 0, -9, 9, 3, -2),
        (5, 0, -9, 9, 3, 3),
    ],
)
def test_int_mutagen(
    seed_value: int,
    value: int,
    min_value: int,
    max_value: int,
    deviation_weight: int,
    expected: int,
):
    seed(seed_value)

    mutagen = IntMutagen(
        value=value,
        min_value=min_value,
        max_value=max_value,
        deviation_weight=deviation_weight,
    )

    mutagen.mutate(1)

    assert mutagen.value == expected


@mark.parametrize(
    'seed_value, value, min_value, max_value, deviation_weight, expected',
    # should bottom out at 0 due to being unsigned
    [
        # should vary up or down by exactly 1
        (0, 0, -1, 1, 1, 1),
        (1, 0, -1, 1, 1, 0),
        (2, 0, -1, 1, 1, 0),
        (3, 0, -1, 1, 1, 0),
        (4, 0, -1, 1, 1, 0),
        (5, 0, -1, 1, 1, 1),

        # should vary up or down by exactly 1 instead of 1-9 due to min and max limits
        (0, 0, -1, 1, 9, 1),
        (1, 0, -1, 1, 9, 0),
        (2, 0, -1, 1, 9, 0),
        (3, 0, -1, 1, 9, 0),
        (4, 0, -1, 1, 9, 0),
        (5, 0, -1, 1, 9, 1),

        # should vary up or down by exactly 1 instead of by 1-9 because deviation_weight is only 1
        (0, 0, -9, 9, 1, 1),
        (1, 0, -9, 9, 1, 0),
        (2, 0, -9, 9, 1, 0),
        (3, 0, -9, 9, 1, 0),
        (4, 0, -9, 9, 1, 0),
        (5, 0, -9, 9, 1, 1),

        # should vary up or down by 1-3 instead of by 1-9 because deviation_weight is only 3
        (0, 0, -9, 9, 3, 2),
        (1, 0, -9, 9, 3, 0),
        (2, 0, -9, 9, 3, 0),
        (3, 0, -9, 9, 3, 0),
        (4, 0, -9, 9, 3, 0),
        (5, 0, -9, 9, 3, 3),
    ],
)
def test_uint_mutagen(
    seed_value: int,
    value: int,
    min_value: int,
    max_value: int,
    deviation_weight: int,
    expected: int,
):
    seed(seed_value)

    mutagen = UintMutagen(
        value=value,
        min_value=min_value,
        max_value=max_value,
        deviation_weight=deviation_weight,
    )

    mutagen.mutate(1)

    assert mutagen.value == expected


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
    ],
)
def test_dna_dimension_size(min_neurons: int, n_dimension: int, expected: int):
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
    ],
)
def test_dna_n_neuron(min_neurons: int, n_dimension: int, expected: int):
    dna = Dna(
        min_neurons=min_neurons,
        n_dimension=n_dimension,
    )

    assert dna.n_neuron == expected
