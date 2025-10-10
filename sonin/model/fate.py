# Cell fating differentiates neurons. We only concern ourselves with neurons since we can ignore biological hardware.
# We need to represent not only the developing cells, but also the environment they develop within.

from dataclasses import dataclass
from typing import Callable, Self

from sonin.model.signal import Level, Signal

type Threshold = int
type IsLower = bool

# Represents a collection of predicates:
#     "does the level of this `Signal` reach the `Threshold` from the direction indicated by `IsLower`?"
type IsLeft = list[(Signal, Threshold, IsLower)]


class FateNode:
    def get_fate(self, signals: dict[Signal, Level]) -> "Fate":
        raise NotImplementedError("FateNode.get_fate")

    def size(self) -> int:
        raise NotImplementedError("FateNode.size")


@dataclass
class Fate(FateNode):
    """ The final configuration of a cell """
    excites: bool
    activation_level: int
    refactory_period: int
    stimulation_amount: int
    stimulation_restore_rate: int
    stimulation_restore_damper: int
    tetanic_threshold: int
    tetanic_activations: int
    tetanic_gap: int

    def get_fate(self, signals: dict[Signal, Level]) -> Self:
        return self

    def size(self) -> int:
        return 1


@dataclass
class BinaryFate(FateNode):
    """ A branch in a decision tree """
    left: FateNode
    right: FateNode
    is_left: IsLeft

    def __iter__(self):
        if isinstance(self.left, BinaryFate):
            yield from iter(self.left)
        else:
            yield self.left

        if isinstance(self.right, BinaryFate):
            yield from iter(self.right)
        else:
            yield self.right

    def get_fate(self, signals: dict[Signal, Level]) -> Fate:
        if all(
            signals[signal] >= level if is_lower else signals[signal] <= level
            for signal, level, is_lower in self.is_left
        ):
            return self.left.get_fate(signals)
        else:
            return self.right.get_fate(signals)

    def size(self) -> int:
        return self.left.size() + self.right.size()


@dataclass
class FateTree:
    root: FateNode | None = None

    def get_fate(self, signals: dict[Signal, Level]) -> Fate | None:
        return self.root.get_fate(signals) if self.root is not None else None

    def find_child_and_parents(self, is_next_left: Callable[[], bool]):
        child = self.root
        parent = None
        g_parent = None
        was_left = None
        g_was_left = None

        if child is None:
            return None, None, None, None, None
        elif isinstance(child, Fate):
            return child, None, None, None, None

        while isinstance(child, BinaryFate):
            g_parent = parent
            parent = child
            g_was_left = was_left
            was_left = is_next_left()

            if was_left:
                child = parent.left
            else:
                child = parent.right

        return child, parent, g_parent, was_left, g_was_left

    def add(self, is_left: IsLeft, leaf: Fate, is_next_left: Callable[[], bool]):
        if self.root is None:
            self.root = leaf
            return

        child, parent, _, was_left, _ = self.find_child_and_parents(is_next_left)

        def left_right(l, r):
            if is_next_left():
                return l, r
            else:
                return r, l

        if child is self.root:
            left, right = left_right(leaf, self.root)

            self.root = BinaryFate(
                left=left,
                right=right,
                is_left=is_left,
            )
        elif was_left:
            left, right = left_right(leaf, parent.left)

            parent.left = BinaryFate(
                left=left,
                right=right,
                is_left=is_left,
            )
        else:
            left, right = left_right(leaf, parent.right)

            parent.right = BinaryFate(
                left=left,
                right=right,
                is_left=is_left,
            )

    def remove(self, is_next_left: Callable[[], bool]):
        if self.root is None:
            return
        elif isinstance(self.root, Fate):
            self.root = None
            return

        _, parent, g_parent, was_left, g_was_left = self.find_child_and_parents(is_next_left)

        if parent is self.root:
            if was_left:
                self.root = parent.right
            else:
                self.root = parent.left
        elif g_was_left:
            if was_left:
                g_parent.left = parent.right
            else:
                g_parent.left = parent.left
        else:
            if was_left:
                g_parent.right = parent.right
            else:
                g_parent.right = parent.left
