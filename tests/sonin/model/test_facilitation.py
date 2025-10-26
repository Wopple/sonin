from pytest import mark

from sonin.model.facilitation import Facilitation


@mark.parametrize(
    "granularity, limit, modulations, expected_up, expected_down",
    [
        (1, 0, [], 1, 1),
        (1, 0, [1], 1, 1),
        (1, 0, [1, -1], 1, 1),
        (1, 0, [1, -1, -1], 1, 1),
        (1, 0, [-1], 1, 1),
        (1, 0, [2], 1, 1),
        (1, 0, [2, -2], 1, 1),
        (1, 0, [2, -3], 1, 1),
        (1, 1, [], 1, 1),
        (1, 1, [1], 2, 1),
        (1, 1, [1, -1], 1, 1),
        (1, 1, [1, -1, -1], 1, 2),
        (1, 1, [-1], 1, 2),
        (1, 1, [2], 2, 1),
        (1, 1, [2, -2], 1, 2),
        (1, 1, [2, -3], 1, 2),
        (2, 1, [], 2, 2),
        (2, 1, [1], 3, 2),
        (2, 1, [1, -1], 2, 2),
        (2, 1, [1, -1, -1], 2, 3),
        (2, 1, [-1], 2, 3),
        (2, 1, [2], 3, 2),
        (2, 1, [2, -2], 2, 3),
        (2, 1, [2, -3], 2, 3),
        (2, 2, [], 2, 2),
        (2, 2, [1], 3, 2),
        (2, 2, [1, -1], 2, 2),
        (2, 2, [1, -1, -1], 2, 3),
        (2, 2, [-1], 2, 3),
        (2, 2, [2], 4, 2),
        (2, 2, [2, -2], 2, 2),
        (2, 2, [2, -3], 2, 3),
        (2, 2, [-1, 2], 3, 2),
        (2, 2, [-1, 3], 4, 2),
        (2, 2, [-2, 5], 4, 2),
    ],
)
def test_facilitation_gear(granularity: int, limit: int, modulations: list[int], expected_up: int, expected_down: int):
    facilitation = Facilitation(
        granularity=granularity,
        limit=limit,
    )

    for m in modulations:
        facilitation.modulate(m)

    assert facilitation.gear.up == expected_up
    assert facilitation.gear.down == expected_down


@mark.parametrize(
    "granularity, limit, modulations, expected",
    [
        (1, 0, [], 0),
        (1, 0, [1], 0),
        (1, 0, [1, -1], 0),
        (1, 0, [1, -1, -1], 0),
        (1, 0, [-1], 0),
        (1, 0, [2], 0),
        (1, 0, [2, -2], 0),
        (1, 0, [2, -3], 0),
        (1, 1, [], 0),
        (1, 1, [1], 1),
        (1, 1, [1, -1], 0),
        (1, 1, [1, -1, -1], -1),
        (1, 1, [-1], -1),
        (1, 1, [2], 1),
        (1, 1, [2, -2], -1),
        (1, 1, [2, -3], -1),
        (2, 1, [], 0),
        (2, 1, [1], 1),
        (2, 1, [1, -1], 0),
        (2, 1, [1, -1, -1], -1),
        (2, 1, [-1], -1),
        (2, 1, [2], 1),
        (2, 1, [2, -2], -1),
        (2, 1, [2, -3], -1),
        (2, 2, [], 0),
        (2, 2, [1], 1),
        (2, 2, [1, -1], 0),
        (2, 2, [1, -1, -1], -1),
        (2, 2, [-1], -1),
        (2, 2, [2], 2),
        (2, 2, [2, -2], 0),
        (2, 2, [2, -3], -1),
        (2, 2, [-1, 2], 1),
        (2, 2, [-1, 3], 2),
        (2, 2, [-2, 5], 2),
    ],
)
def test_facilitation_current(granularity: int, limit: int, modulations: list[int], expected: int):
    facilitation = Facilitation(
        granularity=granularity,
        limit=limit,
    )

    for m in modulations:
        facilitation.modulate(m)

    assert facilitation.current == expected
