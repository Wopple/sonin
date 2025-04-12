from unittest.mock import Mock

from pytest import mark

from sonin.model.fate import FateTree, Fate, Threshold, BinaryFate, FateNode
from sonin.model.signal import Signal


def fate(
    excites: bool = True,
    activation_level: int = 1,
    refactory_period: int = 0,
    stimulation_amount: int = 1,
    stimulation_restore_rate: int = 2,
    stimulation_restore_scalar: int = 1,
    tetanic_threshold: int = 0,
    tetanic_activations: int = 0,
    tetanic_gap: int = 0,
):
    return Fate(
        excites=excites,
        activation_level=activation_level,
        refactory_period=refactory_period,
        stimulation_amount=stimulation_amount,
        stimulation_restore_rate=stimulation_restore_rate,
        stimulation_restore_scalar=stimulation_restore_scalar,
        tetanic_threshold=tetanic_threshold,
        tetanic_activations=tetanic_activations,
        tetanic_gap=tetanic_gap,
    )

@mark.parametrize(
    "adds, expected",
    [
        (
            [([], fate(activation_level=1), [])],
            fate(activation_level=1),
        ),
        (
            [
                ([], fate(activation_level=1), []),
                ([(1, 1)], fate(activation_level=2), [True]),
            ],
            BinaryFate(fate(activation_level=2), fate(activation_level=1), [(1, 1)]),
        ),
        (
            [
                ([], fate(activation_level=1), []),
                ([(1, 1)], fate(activation_level=2), [True]),
                ([(2, 1)], fate(activation_level=3), [False, True]),
            ],
            BinaryFate(
                fate(activation_level=2),
                BinaryFate(fate(activation_level=3), fate(activation_level=1), [(2, 1)]),
                [(1, 1)],
            ),
        ),
    ],
)
def test_add(adds: list[tuple[list[(Signal, Threshold)], Fate, list[bool]]], expected: Fate | BinaryFate):
    tree = FateTree()

    for is_left, leaf, is_next_left in adds:
        mock_is_next_left = Mock()
        mock_is_next_left.side_effect = is_next_left
        tree.add(is_left, leaf, mock_is_next_left)

    assert tree.root.size() == len(adds)
    assert tree.root == expected

@mark.parametrize(
    "root, is_next_left, expected",
    [
        (None, [True], None),
        (None, [False], None),
        (fate(activation_level=1), [True], None),
        (fate(activation_level=1), [False], None),
        (BinaryFate(fate(activation_level=1), fate(activation_level=2), [(1, 1)]), [True], fate(activation_level=2)),
        (BinaryFate(fate(activation_level=1), fate(activation_level=2), [(1, 1)]), [False], fate(activation_level=1)),
        (
            BinaryFate(
                fate(activation_level=1),
                BinaryFate(fate(activation_level=2), fate(activation_level=3), [(1, 1)]),
                [(1, 1)],
            ),
            [True],
            BinaryFate(fate(activation_level=2), fate(activation_level=3), [(1, 1)]),
        ),
        (
            BinaryFate(
                fate(activation_level=1),
                BinaryFate(fate(activation_level=2), fate(activation_level=3), [(1, 1)]),
                [(1, 1)],
            ),
            [False, True],
            BinaryFate(fate(activation_level=1), fate(activation_level=3), [(1, 1)]),
        ),
        (
            BinaryFate(
                fate(activation_level=1),
                BinaryFate(fate(activation_level=2), fate(activation_level=3), [(1, 1)]),
                [(1, 1)],
            ),
            [False, False],
            BinaryFate(fate(activation_level=1), fate(activation_level=2), [(1, 1)]),
        ),
    ],
)
def test_remove(root: FateNode, is_next_left: list[bool], expected: Fate | BinaryFate):
    tree = FateTree(root)
    mock_is_next_left = Mock()
    mock_is_next_left.side_effect = is_next_left
    tree.remove(mock_is_next_left)

    assert tree.root == expected
