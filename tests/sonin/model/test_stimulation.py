from pytest import mark

from sonin.model.stimulation import SnapBack


@mark.parametrize(
    'baseline, rate, scalar, initial, expected',
    [
        (0, 2, 1, 0, 0),
        (0, 2, 1, 1, 0),
        (0, 2, 1, -1, 0),
        (-1, 2, 1, -2, -1),
        (-2, 2, 1, -2, -2),
        (0, 3, 2, 10, 6),
        (3, 3, 2, 10, 7),
        (-3, 3, 2, 10, 5),
        (0, 3, 2, -10, -6),
        (3, 3, 2, -10, -5),
        (-3, 3, 2, -10, -7),
    ]
)
def test_snap_back_step(baseline: int, rate: int, scalar: int, initial: int, expected: int):
    snap_back = SnapBack(baseline, rate, scalar)

    snap_back.value = initial
    snap_back.step()

    assert snap_back.value == expected
