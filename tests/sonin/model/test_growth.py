from sonin.model.growth import Incubator
from sonin.model.hypercube import Vector
from sonin.model.signal import SignalProfile


# since there is only 1 cell, nothing should change
def test_growth_1():
    dimension_size = 1

    incubator = Incubator(
        n_dimension=2,
        dimension_size=dimension_size,
        environment=[(1, 20, Vector(value=(0, 0), dimension_size=dimension_size))],
        signal_profile=SignalProfile({
            1: {1: 5, 2: -10},
            2: {1: 0},
        }),
    )

    incubator.initialize({
        1: 1,
        2: 10,
    })

    incubator.incubate()

    assert incubator.cells.get([0, 0]).signals == {1: 1, 2: 10}


# since the center contains all the cells, nothing should change
def test_growth_2():
    dimension_size = 2

    incubator = Incubator(
        n_dimension=2,
        dimension_size=dimension_size,
        environment=[(1, 20, Vector(value=(0, 0), dimension_size=dimension_size))],
        signal_profile=SignalProfile({
            1: {1: 5, 2: -10},
            2: {1: 0},
        }),
    )

    incubator.initialize({
        1: 1,
        2: 10,
    })

    incubator.incubate()

    assert incubator.cells.get([0, 0]).signals == {1: 1, 2: 10}
    assert incubator.cells.get([0, 1]).signals == {1: 1, 2: 10}
    assert incubator.cells.get([1, 0]).signals == {1: 1, 2: 10}
    assert incubator.cells.get([1, 1]).signals == {1: 1, 2: 10}


# this should pull more signals closer to [0, 0] and fewer signals closer to [2, 2] 
def test_growth_3():
    dimension_size = 3

    incubator = Incubator(
        n_dimension=2,
        dimension_size=dimension_size,
        environment=[
            (1, 4, Vector(value=(0, 0), dimension_size=dimension_size)),
            (1, 1, Vector(value=(2, 2), dimension_size=dimension_size)),
        ],
        signal_profile=SignalProfile({
            2: {1: 1},
        }),
    )

    incubator.initialize({
        2: 3 * 3 * 16,
    })

    incubator.incubate()

    assert incubator.cells.get([0, 0]).signals == {2: 104}
    assert incubator.cells.get([0, 1]).signals == {2: 5}
    assert incubator.cells.get([0, 2]).signals == {2: 1}
    assert incubator.cells.get([1, 0]).signals == {2: 5}
    assert incubator.cells.get([1, 1]).signals == {2: 2}
    assert incubator.cells.get([1, 2]).signals == {2: 7}
    assert incubator.cells.get([2, 0]).signals == {2: 1}
    assert incubator.cells.get([2, 1]).signals == {2: 7}
    assert incubator.cells.get([2, 2]).signals == {2: 12}
