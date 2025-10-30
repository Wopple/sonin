from pytest import mark

from sonin.model.dna import Dna
from sonin.model.fate import FateTree
from sonin.model.hypercube import CityShape, CubeShape, Hypercube, Vector
from sonin.model.signal import SignalProfile

DIMENSION_SIZE = 50


def dna(n_dimension: int, dimension_size: int) -> Dna:
    return Dna(
        n_dimension=n_dimension,
        dimension_size=dimension_size,
        n_synapse=1,
        activation_level=1,
        max_neuron_strength=1,
        axon_range=1,
        refactory_period=1,
        environment=[],
        incubation_signals={},
        signal_profile=SignalProfile(affinities={}),
        fate_tree=FateTree(),
    )


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


@mark.parametrize(
    "center, size, expected",
    [
        (Vector.of([0], 10), 1, [(0,)]),
        (Vector.of([0, 0], 10), 1, [(0, 0)]),
        (Vector.of([0, 0, 0], 10), 1, [(0, 0, 0)]),

        (Vector.of([0], 10), 2, [(0,), (1,)]),

        (Vector.of([0, 0], 10), 2, [
            (0, 0),
            (1, 0),
            (0, 1),
            (1, 1),
        ]),

        # [][] [][]
        # CT[] [][]
        (Vector.of([0, 0, 0], 10), 2, [
            (x, y, z)
            for z in range(2)
            for y in range(2)
            for x in range(2)
        ]),

        # __[][] __[]CT
        # __[][] __[][]
        # ______ ______
        (Vector.of([2, 2, 2], 3), 2, [
            (x + 1, y + 1, z + 1)
            for z in range(2)
            for y in range(2)
            for x in range(2)
        ]),

        (Vector.of([0], 10), 3, [(0,), (1,), (2,)]),

        # [][][]
        # [][][]
        # CT[][]
        (Vector.of([0, 0], 10), 3, [
            (x, y)
            for y in range(3)
            for x in range(3)
        ]),

        # [][][] [][][] [][][]
        # [][][] [][][] [][][]
        # CT[][] [][][] [][][]
        (Vector.of([0, 0, 0], 10), 3, [
            (x, y, z)
            for z in range(3)
            for y in range(3)
            for x in range(3)
        ]),

        # []CT[][]
        (Vector.of([1], 10), 3, [(0,), (1,), (2,), (3,)]),

        # [][][][]
        # [][][][]
        # []CT[][]
        (Vector.of([1, 0], 10), 3, [
            (x, y)
            for y in range(3)
            for x in range(4)
        ]),

        # [][][][] [][][][] [][][][] [][][][]
        # [][][][] [][][][] [][][][] [][][][]
        # [][][][] []CT[][] [][][][] [][][][]
        (Vector.of([1, 0, 0], 10), 3, [
            (x, y, z)
            for z in range(3)
            for y in range(3)
            for x in range(4)
        ]),

        # ____________ __[][][][][] __[][][][][] __[][][][][] __[][][][][] __[][][][][]
        # ____________ __[][][][][] __[][][][][] __[][][][][] __[][][][][] __[][][][][]
        # ____________ __[][][][][] __[][][][][] __[][]CT[][] __[][][][][] __[][][][][]
        # ____________ __[][][][][] __[][][][][] __[][][][][] __[][][][][] __[][][][][]
        # ____________ __[][][][][] __[][][][][] __[][][][][] __[][][][][] __[][][][][]
        # ____________ ____________ ____________ ____________ ____________ ____________
        (Vector.of([3, 3, 3], 10), 3, [
            (x + 1, y + 1, z + 1)
            for z in range(5)
            for y in range(5)
            for x in range(5)
        ]),
    ],
)
def test_cube_shape(center: Vector, size: int, expected: list[tuple[int, ...]]):
    shape = CubeShape(center=center, size=size)
    actual = [position.value for position in shape.positions()]

    assert actual == expected


@mark.parametrize(
    "center, size, expected",
    [
        (Vector.of([0], 10), 1, [(0,)]),
        (Vector.of([0, 0], 10), 1, [(0, 0)]),
        (Vector.of([0, 0, 0], 10), 1, [(0, 0, 0)]),

        (Vector.of([0], 10), 2, [(0,), (1,)]),

        (Vector.of([0, 0], 10), 2, [
            (0, 0),
            (1, 0),
            (0, 1),
        ]),

        # []__ ____
        # CT[] []__
        (Vector.of([0, 0, 0], 10), 2, [
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
        ]),

        # __[] []CT
        # ____ __[]
        (Vector.of([1, 1, 1], 2), 2, [
            (1, 1, 0),
            (1, 0, 1),
            (0, 1, 1),
            (1, 1, 1),
        ]),

        (Vector.of([0], 10), 3, [(0,), (1,), (2,)]),

        # []____
        # [][]__
        # CT[][]
        (Vector.of([0, 0], 10), 3, [
            (0, 0),
            (1, 0),
            (2, 0),
            (0, 1),
            (1, 1),
            (0, 2),
        ]),

        # []____ ______ ______
        # [][]__ []____ ______
        # CT[][] [][]__ []____
        (Vector.of([0, 0, 0], 10), 3, [
            (0, 0, 0),
            (1, 0, 0),
            (2, 0, 0),
            (0, 1, 0),
            (1, 1, 0),
            (0, 2, 0),
            (0, 0, 1),
            (1, 0, 1),
            (0, 1, 1),
            (0, 0, 2),
        ]),

        # []CT[][]
        (Vector.of([1], 10), 3, [(0,), (1,), (2,), (3,)]),

        # __[]____
        # [][][]__
        # []CT[][]
        (Vector.of([1, 0], 10), 3, [
            (0, 0),
            (1, 0),
            (2, 0),
            (3, 0),
            (0, 1),
            (1, 1),
            (2, 1),
            (1, 2),
        ]),

        # __[]____ ________ ________
        # [][][]__ __[]____ ________
        # []CT[][] [][][]__ __[]____
        (Vector.of([1, 0, 0], 10), 3, [
            (0, 0, 0),
            (1, 0, 0),
            (2, 0, 0),
            (3, 0, 0),
            (0, 1, 0),
            (1, 1, 0),
            (2, 1, 0),
            (1, 2, 0),
            (0, 0, 1),
            (1, 0, 1),
            (2, 0, 1),
            (1, 1, 1),
            (1, 0, 2),
        ]),

        # ____________ ____________ ____________ ______[]____ ____________ ____________
        # ____________ ____________ ______[]____ ____[][][]__ ______[]____ ____________
        # ____________ ______[]____ ____[][][]__ __[][]CT[][] ____[][][]__ ______[]____
        # ____________ ____________ ______[]____ ____[][][]__ ______[]____ ____________
        # ____________ ____________ ____________ ______[]____ ____________ ____________
        # ____________ ____________ ____________ ____________ ____________ ____________
        (Vector.of([3, 3, 3], 10), 3, [
            (3, 3, 1),
            (3, 2, 2),
            (2, 3, 2),
            (3, 3, 2),
            (4, 3, 2),
            (3, 4, 2),
            (3, 1, 3),
            (2, 2, 3),
            (3, 2, 3),
            (4, 2, 3),
            (1, 3, 3),
            (2, 3, 3),
            (3, 3, 3),
            (4, 3, 3),
            (5, 3, 3),
            (2, 4, 3),
            (3, 4, 3),
            (4, 4, 3),
            (3, 5, 3),
            (3, 2, 4),
            (2, 3, 4),
            (3, 3, 4),
            (4, 3, 4),
            (3, 4, 4),
            (3, 3, 5),
        ]),
    ],
)
def test_city_shape(center: Vector, size: int, expected: list[tuple[int, ...]]):
    shape = CityShape(center=center, size=size)
    actual = [position.value for position in shape.positions()]

    assert actual == expected


def test_initialize():
    test_dna = dna(3, 3)

    hypercube = Hypercube(
        n_dimension=test_dna.n_dimension,
        dimension_size=test_dna.dimension_size,
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


@mark.parametrize(
    'dna, expected_center',
    [
        (dna(1, 1), [(0,)]),
        (dna(1, 2), [(0,), (1,)]),
        (dna(1, 3), [(1,)]),
        (dna(1, 4), [(1,), (2,)]),

        (dna(2, 1), [(0, 0)]),
        (dna(2, 2), [(0, 0), (0, 1), (1, 0), (1, 1)]),
        (dna(2, 3), [(1, 1)]),
        (dna(2, 4), [(1, 1), (1, 2), (2, 1), (2, 2)]),

        (dna(3, 1), [(0, 0, 0)]),
        (dna(3, 2), [
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, 0),
            (0, 1, 1),
            (1, 0, 0),
            (1, 0, 1),
            (1, 1, 0),
            (1, 1, 1),
        ]),
        (dna(3, 3), [(1, 1, 1)]),
        (dna(3, 4), [
            (1, 1, 1),
            (1, 1, 2),
            (1, 2, 1),
            (1, 2, 2),
            (2, 1, 1),
            (2, 1, 2),
            (2, 2, 1),
            (2, 2, 2),
        ]),
    ]
)
def test_center(dna: Dna, expected_center: list[tuple[int, ...]]):
    hypercube = Hypercube(
        n_dimension=dna.n_dimension,
        dimension_size=dna.dimension_size,
    )

    hypercube.initialize(lambda position: position.value)

    assert hypercube.center() == expected_center
