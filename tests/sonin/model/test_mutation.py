from pytest import mark

from sonin.model.mutation import BoolMutagen, IntMutagen, OptionalMutagen, UintMutagen
from sonin.sonin_random import seed


@mark.parametrize(
    'value, deviation_weight',
    [
        (True, 1),
        (True, 2),
        (True, 10),
        (True, 100),
        (False, 1),
        (False, 2),
        (False, 10),
        (False, 100),
    ],
)
def test_bool_mutagen(
    value: bool,
    deviation_weight: int,
):
    mutagen = BoolMutagen(
        bool_value=value,
        deviation_weight=deviation_weight,
    )

    mutagen.mutate(1)

    assert mutagen.value is not value


@mark.parametrize(
    'seed_value, value, min_value, max_value, deviation_weight, expected',
    [
        # should vary up or down by exactly 1
        (0, 0, -1, 1, 1, -1),
        (1, 0, -1, 1, 1, 1),
        (2, 0, -1, 1, 1, 1),
        (3, 0, -1, 1, 1, 1),
        (4, 0, -1, 1, 1, -1),
        (5, 0, -1, 1, 1, 1),

        # should vary up or down by exactly 1 instead of 1-9 due to min and max limits
        (0, 0, -1, 1, 9, -1),
        (1, 0, -1, 1, 9, 1),
        (2, 0, -1, 1, 9, 1),
        (3, 0, -1, 1, 9, 1),
        (4, 0, -1, 1, 9, -1),
        (5, 0, -1, 1, 9, 1),

        # should vary up or down by exactly 1 instead of by 1-9 because deviation_weight is only 1
        (0, 0, -9, 9, 1, -1),
        (1, 0, -9, 9, 1, 1),
        (2, 0, -9, 9, 1, 1),
        (3, 0, -9, 9, 1, 1),
        (4, 0, -9, 9, 1, -1),
        (5, 0, -9, 9, 1, 1),

        # should vary up or down by 1-3 instead of by 1-9 because deviation_weight is only 3
        (0, 0, -9, 9, 3, -2),
        (1, 0, -9, 9, 3, 2),
        (2, 0, -9, 9, 3, 3),
        (3, 0, -9, 9, 3, 1),
        (4, 0, -9, 9, 3, -1),
        (5, 0, -9, 9, 3, 2),
    ],
)
def test_int_mutagen(
    seed_value: int,
    value: int,
    min_value: int,
    max_value: int,
    deviation_weight: int,
    expected: int,
):
    seed(seed_value)

    mutagen = IntMutagen(
        int_value=value,
        min_value=min_value,
        max_value=max_value,
        deviation_weight=deviation_weight,
    )

    mutagen.mutate(1)

    assert mutagen.value == expected


@mark.parametrize(
    'seed_value, value, min_value, max_value, deviation_weight, expected',
    # should bottom out at 0 due to being unsigned
    [
        # should vary up or down by exactly 1
        (0, 0, -1, 1, 1, 0),
        (1, 0, -1, 1, 1, 1),
        (2, 0, -1, 1, 1, 1),
        (3, 0, -1, 1, 1, 1),
        (4, 0, -1, 1, 1, 0),
        (5, 0, -1, 1, 1, 1),

        # should vary up or down by exactly 1 instead of 1-9 due to min and max limits
        (0, 0, -1, 1, 9, 0),
        (1, 0, -1, 1, 9, 1),
        (2, 0, -1, 1, 9, 1),
        (3, 0, -1, 1, 9, 1),
        (4, 0, -1, 1, 9, 0),
        (5, 0, -1, 1, 9, 1),

        # should vary up or down by exactly 1 instead of by 1-9 because deviation_weight is only 1
        (0, 0, -9, 9, 1, 0),
        (1, 0, -9, 9, 1, 1),
        (2, 0, -9, 9, 1, 1),
        (3, 0, -9, 9, 1, 1),
        (4, 0, -9, 9, 1, 0),
        (5, 0, -9, 9, 1, 1),

        # should vary up or down by 1-3 instead of by 1-9 because deviation_weight is only 3
        (0, 0, -9, 9, 3, 0),
        (1, 0, -9, 9, 3, 2),
        (2, 0, -9, 9, 3, 3),
        (3, 0, -9, 9, 3, 1),
        (4, 0, -9, 9, 3, 0),
        (5, 0, -9, 9, 3, 2),
    ],
)
def test_uint_mutagen(
    seed_value: int,
    value: int,
    min_value: int,
    max_value: int,
    deviation_weight: int,
    expected: int,
):
    seed(seed_value)

    mutagen = UintMutagen(
        int_value=value,
        min_value=min_value,
        max_value=max_value,
        deviation_weight=deviation_weight,
    )

    mutagen.mutate(1)

    assert mutagen.value == expected


@mark.parametrize(
    'seed_value, value, exists, expected',
    [
        # seed 1 chooses the exist mutagen
        (1, True, True, None),
        (1, False, True, None),

        # seed 2 chooses the value mutagen
        (2, True, True, False),
        (2, False, True, True),

        # always bring the value into existence unchanged
        (0, True, False, True),
        (0, False, False, False),
        (1, True, False, True),
        (1, False, False, False),
    ],
)
def test_optional_mutagen(
    seed_value: int,
    value: bool,
    exists: bool,
    expected: bool | None,
):
    seed(seed_value)

    value_mutagen = BoolMutagen(bool_value=value)
    exists_mutagen = BoolMutagen(bool_value=exists)

    optional_mutagen = OptionalMutagen[bool](
        mutagen=value_mutagen,
        exists=exists_mutagen,
    )

    optional_mutagen.mutate(1)

    assert optional_mutagen.value is expected
