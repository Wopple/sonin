from pytest import mark

from sonin.model.gear import Gear


@mark.parametrize(
    "up, down, io",
    [
        (1, 1, [(1, 1), (2, 2), (3, 3), (4, 4)]),
        (2, 1, [(1, 2), (2, 4), (3, 6), (4, 8)]),
        (1, 2, [(1, 0), (2, 1), (1, 1), (4, 2)]),
        (2, 2, [(1, 1), (2, 2), (3, 3), (4, 4)]),
        (2, 3, [(1, 0), (1, 1), (1, 1), (1, 0)]),
        (2, 3, [(2, 1), (2, 1), (2, 2), (2, 1)]),
    ],
)
def test_gear(up: int, down: int, io: list[tuple[int, int]]):
    gear = Gear(
        up=up,
        down=down,
    )

    for x, expected in io:
        assert gear(x) == expected
