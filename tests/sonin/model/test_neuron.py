from pytest import mark

from sonin.model.hypercube import Vector
from sonin.model.neuron import Axon, TetanicPeriod


@mark.parametrize(
    'threshold, activations, gap, num_steps, expected',
    [
        (2, 2, 0, 1, False),
        (2, 2, 0, 2, False),
        (2, 2, 0, 3, True),  # active
        (2, 2, 0, 4, True),
        (2, 2, 0, 5, False),  # inactive
        (2, 2, 0, 6, False),
        (2, 2, 0, 7, True),

        (2, 2, 1, 1, False),
        (2, 2, 1, 2, False),
        (2, 2, 1, 3, True),  # active
        (2, 2, 1, 4, False),  # gap
        (2, 2, 1, 5, True),
        (2, 2, 1, 6, False),  # gap
        (2, 2, 1, 7, False),  # inactive
        (2, 2, 1, 8, False),
        (2, 2, 1, 9, True),

        (2, 1, 1, 1, False),
        (2, 1, 1, 2, False),
        (2, 1, 1, 3, True),  # active
        (2, 1, 1, 4, False),  # gap
        (2, 1, 1, 5, False),  # inactive
        (2, 1, 1, 6, False),
        (2, 1, 1, 7, True),
    ],
)
def test_tetanic_period(threshold: int, activations: int, gap: int, num_steps: int, expected: bool):
    tetanic_period = TetanicPeriod(
        enabled=True,
        threshold=threshold,
        activations=activations,
        gap=gap,
    )

    c_time: int = 0

    for i in range(num_steps):
        c_time = i
        tetanic_period.step(c_time)

    assert tetanic_period.is_active(c_time) is expected

