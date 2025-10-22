from pytest import mark

from sonin.model.dna import Dna
from sonin.model.hypercube import Hypercube, Vector

DIMENSION_SIZE = 50


@mark.parametrize(
    "a, b, expected",
    [
        ((0,), (0,), (0,)),
        ((1,), (0,), (1,)),
        ((0,), (1,), (1,)),
        ((1,), (1,), (2,)),
        ((1,), (-1,), (0,)),
        ((-1,), (1,), (0,)),
        ((-1,), (-1,), (-2,)),
        ((1, 2), (3, 4), (4, 6)),
    ],
)
def test_add(a: tuple[int, ...], b: tuple[int, ...], expected: tuple[int, ...]):
    v_a = Vector(value=a, dimension_size=len(a))
    v_b = Vector(value=b, dimension_size=len(b))
    v_expected = Vector(value=expected, dimension_size=len(expected))

    assert v_a + b == v_expected
    assert v_a + list(b) == v_expected
    assert v_a + v_b == v_expected


@mark.parametrize(
    "a, b, expected",
    [
        ((0,), (0,), (0,)),
        ((1,), (0,), (1,)),
        ((0,), (1,), (-1,)),
        ((1,), (1,), (0,)),
        ((1,), (-1,), (2,)),
        ((-1,), (1,), (-2,)),
        ((-1,), (-1,), (0,)),
        ((1, 2), (3, 4), (-2, -2)),
    ],
)
def test_sub(a: tuple[int, ...], b: tuple[int, ...], expected: tuple[int, ...]):
    v_a = Vector(value=a, dimension_size=DIMENSION_SIZE)
    v_b = Vector(value=b, dimension_size=DIMENSION_SIZE)
    v_expected = Vector(value=expected, dimension_size=DIMENSION_SIZE)

    assert v_a - v_b == v_expected


@mark.parametrize(
    "a, b, expected",
    [
        ((0,), 0, (0,)),
        ((1,), 0, (0,)),
        ((0,), 1, (0,)),
        ((1,), 1, (1,)),
        ((1,), -1, (-1,)),
        ((-1,), -1, (1,)),
        ((2,), 3, (6,)),
        ((-3,), 2, (-6,)),
        ((1, 2), 3, (3, 6)),
    ],
)
def test_mul(a: tuple[int, ...], b: int, expected: tuple[int, ...]):
    v_a = Vector(value=a, dimension_size=DIMENSION_SIZE)
    v_expected = Vector(value=expected, dimension_size=DIMENSION_SIZE)

    assert v_a * b == v_expected


@mark.parametrize(
    "a, b, expected",
    [
        ((0,), (0,), 0),
        ((1,), (0,), 0),
        ((0,), (1,), 0),
        ((1,), (1,), 1),
        ((1,), (-1,), -1),
        ((-1,), (1,), -1),
        ((1, 1), (-1, -1), -2),
        ((1, 2), (3, 4), 11),
    ],
)
def test_dot_product(a: tuple[int, ...], b: tuple[int, ...], expected: int):
    v_a = Vector(value=a, dimension_size=DIMENSION_SIZE)
    v_b = Vector(value=b, dimension_size=DIMENSION_SIZE)

    assert v_a * v_b == expected


@mark.parametrize(
    "a, b, expected",
    [
        ((0,), 1, (0,)),
        ((1,), 1, (1,)),
        ((2,), 1, (2,)),
        ((3,), 2, (1,)),
        ((4,), 2, (2,)),
        ((-4,), 2, (-2,)),
        ((4,), -2, (-2,)),
        ((2,), 3, (0,)),
        ((-3,), 2, (-1,)),
        ((3, 6), 3, (1, 2)),
    ],
)
def test_div(a: tuple[int, ...], b: int, expected: tuple[int, ...]):
    v_a = Vector(value=a, dimension_size=DIMENSION_SIZE)
    v_expected = Vector(value=expected, dimension_size=DIMENSION_SIZE)

    assert v_a / b == v_expected
    assert v_a // b == v_expected


@mark.parametrize(
    "a, b, expected",
    [
        ((1,), 0, (0,)),
        ((1,), 1, (1,)),
        ((1,), 2, (2,)),
        ((2,), 3, (1,)),
        ((2,), 4, (2,)),
        ((2,), -4, (-2,)),
        ((-2,), 4, (-2,)),
        ((3,), 2, (0,)),
        ((2,), -3, (-1,)),
        ((6, 3), 6, (1, 2)),
    ],
)
def test_rdiv(a: tuple[int, ...], b: int, expected: tuple[int, ...]):
    v_a = Vector(value=a, dimension_size=DIMENSION_SIZE)
    v_expected = Vector(value=expected, dimension_size=DIMENSION_SIZE)

    assert b / v_a == v_expected
    assert b // v_a == v_expected


@mark.parametrize(
    "before, expected",
    [
        ((-1,), True),
        ((0,), False),
        ((1,), False),
        ((DIMENSION_SIZE // 2,), False),
        ((DIMENSION_SIZE - 2,), False),
        ((DIMENSION_SIZE - 1,), False),
        ((DIMENSION_SIZE,), True),
        ((-1, -1), True),
        ((0, -1), True),
        ((-1, 0), True),
        ((0, 0), False),
        ((DIMENSION_SIZE, DIMENSION_SIZE), True),
        ((0, DIMENSION_SIZE), True),
        ((DIMENSION_SIZE, 0), True),
        ((0, 0), False),
    ],
)
def test_out_of_bounds(before: tuple[int, ...], expected: bool):
    actual = Vector(value=before, dimension_size=DIMENSION_SIZE).out_of_bounds()

    assert actual is expected


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
        ((10, 4, 0), (1, 0, 0)),
        ((10, 5, 0), (1, 1, 0)),
        ((4, 10, 0), (0, 1, 0)),
        ((5, 10, 0), (1, 1, 0)),
        ((10, 4, -4), (1, 0, 0)),
        ((10, 5, -4), (1, 1, 0)),
        ((4, 10, -4), (0, 1, 0)),
        ((5, 10, -4), (1, 1, 0)),
        ((10, 4, -5), (1, 0, -1)),
        ((10, 5, -5), (1, 1, -1)),
        ((4, 10, -5), (0, 1, -1)),
        ((5, 10, -5), (1, 1, -1)),
    ],
)
def test_city_unit(before: tuple[int, ...], expected: tuple[int, ...]):
    actual = Vector(value=before, dimension_size=DIMENSION_SIZE).city_unit().value

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
