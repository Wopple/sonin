from pytest import mark

from sonin.model.metric import FrequencyProfile


@mark.parametrize(
    'size, c_times, expected',
    [
        (1, [], -1),
        (1, [1], -1),
        (1, [1, 2], 1),
        (1, [1, 2, 3], 1),
        (1, [1, 2, 3, 5], 2),
        (2, [1, 2], 1),
        (2, [1, 2, 3], 1),
        (2, [1, 2, 3, 5], 1),
        (2, [1, 2, 3, 6], 2),
        (2, [1, 2, 3, 7], 2),
        (2, [1, 2, 3, 8], 3),
    ],
)
def test_frequency_mean(size: int, c_times: list[int], expected: int):
    frequency = FrequencyProfile(size)

    for c_time in c_times:
        frequency.record(c_time)

    assert frequency.mean == expected


@mark.parametrize(
    'size, c_times, expected',
    [
        (1, [], -1),
        (1, [1], -1),
        (1, [1, 2], 0),
        (1, [1, 2, 3], 0),
        (1, [1, 2, 3, 5], 0),
        (2, [1, 2], 0),
        (2, [1, 2, 3], 0),
        (2, [1, 2, 3, 5], 1),
        (2, [1, 2, 3, 6], 2),
        (2, [1, 2, 3, 7], 3),
        (2, [1, 2, 3, 8], 4),
        (2, [1, 2, 3, 9], 5),
        (2, [1, 2, 3, 10], 6),
    ],
)
def test_frequency_instability(size: int, c_times: list[int], expected: int):
    frequency = FrequencyProfile(size)

    for c_time in c_times:
        frequency.record(c_time)

    assert frequency.instability == expected
