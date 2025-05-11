from pytest import mark

from sonin.model.math import div


@mark.parametrize(
    "a, b, expected",
    [
        (3, 2, 1),
        (-3, 2, -1),
        (3, -2, -1),
        (-3, -2, 1),
    ],
)
def test_div(a: int, b: int, expected: int):
    assert div(a, b) == expected
