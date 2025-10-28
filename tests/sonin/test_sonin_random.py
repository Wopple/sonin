from pytest import mark

from sonin.sonin_random import rand_bool, rotate_right_32, seed as set_seed


@mark.parametrize(
    "x, num, expected",
    [
        (
            0b00000000000000000000000000000000,
            1,
            0b00000000000000000000000000000000,
        ),
        (
            0b10000000000000000000000000000000,
            1,
            0b01000000000000000000000000000000,
        ),
        (
            0b00000000000000000000000000000001,
            1,
            0b10000000000000000000000000000000,
        ),
        (
            0b01010101010101010101010101010101,
            1,
            0b10101010101010101010101010101010,
        ),
        (
            0b01010101010101010101010101010101,
            2,
            0b01010101010101010101010101010101,
        ),
        (
            0b01010101010101010101010101010101,
            4,
            0b01010101010101010101010101010101,
        ),
        (
            0x01234567,
            0,
            0x01234567,
        ),
        (
            0x01234567,
            4,
            0x70123456,
        ),
    ],
)
def test_rotate_right_32(x: int, num: int, expected: int):
    assert rotate_right_32(x, num) == expected


@mark.parametrize(
    "seed, expected",
    [
        (1, True),
        (3, True),
        (4, False),
        (5, True),
        (7, True),
        (10, False),
    ],
)
def test_rand_bool(seed: int, expected: int):
    set_seed(seed)

    assert rand_bool() == expected
