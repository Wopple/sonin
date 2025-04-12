from pytest import mark

from sonin.model.dna import Dna
from sonin.model.hypercube import Hypercube, Vector


@mark.parametrize(
    "before, expected",
    [
        ((0,), (0,)),
        ((1,), (1,)),
        ((-1,), (-1,)),
        ((2,), (1,)),
        ((-2,), (-1,)),
        ((0, 0), (0, 0)),
        ((0, 2), (0, 1)),
        ((2, 0), (1, 0)),
        ((2, 2), (1, 1)),
        ((10, 4), (1, 0)),
        ((10, 5), (1, 1)),
        ((4, 10), (0, 1)),
        ((5, 10), (1, 1)),
    ]
)
def test_city_unit(before: tuple[int, ...], expected: tuple[int, ...]):
    actual = Vector(len(before), before).city_unit().value

    assert actual == expected


def test_initialize():
    dna = Dna(
        min_neurons=27,
        n_dimension=3,
    )

    hypercube = Hypercube(
        n_dimension=dna.n_dimension,
        dimension_size=dna.dimension_size,
    )

    hypercube.initialize(lambda position: position.value)

    assert hypercube.items == [
        (0, 0, 0),
        (0, 0, 1),
        (0, 0, 2),
        (0, 1, 0),
        (0, 1, 1),
        (0, 1, 2),
        (0, 2, 0),
        (0, 2, 1),
        (0, 2, 2),
        (1, 0, 0),
        (1, 0, 1),
        (1, 0, 2),
        (1, 1, 0),
        (1, 1, 1),
        (1, 1, 2),
        (1, 2, 0),
        (1, 2, 1),
        (1, 2, 2),
        (2, 0, 0),
        (2, 0, 1),
        (2, 0, 2),
        (2, 1, 0),
        (2, 1, 1),
        (2, 1, 2),
        (2, 2, 0),
        (2, 2, 1),
        (2, 2, 2),
    ]
