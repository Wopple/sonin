from unittest.mock import Mock

from pytest import mark

from sonin.model.fate import BinaryFate, Fate, FateNode, FateTree, IsLeft, Threshold
from sonin.model.neuron import TetanicPeriod
from sonin.model.signal import Signal, SignalCount
from sonin.model.stimulation import SnapBack, Stimulation


def fate(
    excites: bool = True,
    axon_signals: dict[Signal, SignalCount] | None = None,
    activation_level: int = 1,
    refactory_period: int = 0,
    stimulation_amount: int = 1,
    stimulation_baseline: int = 0,
    stimulation_restore_rate: int = 2,
    stimulation_restore_damper: int = 1,
    tetanic_threshold: int = 0,
    tetanic_activations: int = 0,
    tetanic_gap: int = 0,
):
    return Fate(
        excites=excites,
        axon_signals=axon_signals or {},
        activation_level=activation_level,
        refactory_period=refactory_period,
        stimulation=Stimulation(
            amount=stimulation_amount,
            snap_back=SnapBack(
                baseline=stimulation_baseline,
                restore_rate=stimulation_restore_rate,
                restore_damper=stimulation_restore_damper,
            ),
        ),
        tetanic_period=TetanicPeriod(
            threshold=tetanic_threshold,
            activations=tetanic_activations,
            gap=tetanic_gap,
        ),
    )


def binary_fate(
    left: FateNode,
    right: FateNode,
    is_left: IsLeft,
):
    return BinaryFate(
        left=left,
        right=right,
        is_left=is_left,
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
                ([(1, 1, True)], fate(activation_level=2), [True]),
            ],
            binary_fate(fate(activation_level=2), fate(activation_level=1), [(1, 1, True)]),
        ),
        (
            [
                ([], fate(activation_level=1), []),
                ([(1, 1, True)], fate(activation_level=2), [True]),
                ([(2, 1, True)], fate(activation_level=3), [False, True]),
            ],
            binary_fate(
                fate(activation_level=2),
                binary_fate(fate(activation_level=3), fate(activation_level=1), [(2, 1, True)]),
                [(1, 1, True)],
            ),
        ),
    ],
)
def test_add(adds: list[tuple[IsLeft, Fate, list[bool]]], expected: Fate | BinaryFate):
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
        (binary_fate(fate(activation_level=1), fate(activation_level=2), [(1, 1, True)]), [True], fate(activation_level=2)),
        (binary_fate(fate(activation_level=1), fate(activation_level=2), [(1, 1, True)]), [False], fate(activation_level=1)),
        (
            binary_fate(
                fate(activation_level=1),
                binary_fate(fate(activation_level=2), fate(activation_level=3), [(1, 1, True)]),
                [(1, 1, True)],
            ),
            [True],
            binary_fate(fate(activation_level=2), fate(activation_level=3), [(1, 1, True)]),
        ),
        (
            binary_fate(
                fate(activation_level=1),
                binary_fate(fate(activation_level=2), fate(activation_level=3), [(1, 1, True)]),
                [(1, 1, True)],
            ),
            [False, True],
            binary_fate(fate(activation_level=1), fate(activation_level=3), [(1, 1, True)]),
        ),
        (
            binary_fate(
                fate(activation_level=1),
                binary_fate(fate(activation_level=2), fate(activation_level=3), [(1, 1, True)]),
                [(1, 1, True)],
            ),
            [False, False],
            binary_fate(fate(activation_level=1), fate(activation_level=2), [(1, 1, True)]),
        ),
    ],
)
def test_remove(root: FateNode, is_next_left: list[bool], expected: Fate | BinaryFate):
    tree = FateTree(root=root)
    mock_is_next_left = Mock()
    mock_is_next_left.side_effect = is_next_left
    tree.remove(mock_is_next_left)

    assert tree.root == expected

@mark.parametrize(
    "is_left, signals, expected_activation_level",
    [
        ([(1, 0, True)], {}, 1),
        ([(1, 0, False)], {}, 1),
        ([(1, 1, True)], {}, 1),
        ([(1, 1, False)], {}, 2),
        ([(1, 0, True)], {1: 0}, 1),
        ([(1, 0, False)], {1: 0}, 1),
        ([(1, 1, True)], {1: 0}, 1),
        ([(1, 1, False)], {1: 0}, 2),
        ([(1, 2, True), (2, 3, False)], {1: 1, 2: 4}, 1),
        ([(1, 2, True), (2, 3, False)], {1: 1, 2: 4, 3: 7}, 1),
        ([(1, 2, False), (2, 3, False)], {1: 1, 2: 4}, 2),
        ([(1, 2, True), (2, 3, True)], {1: 1, 2: 4}, 2),
    ],
)
def test_binary_fate(
    is_left: list[tuple[Signal, Threshold, bool]],
    signals: dict[Signal, SignalCount],
    expected_activation_level: int,
):
    actual = binary_fate(
        left=fate(activation_level=1),
        right=fate(activation_level=2),
        is_left=is_left,
    ).get_fate(signals)

    assert actual.activation_level == expected_activation_level
