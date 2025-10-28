# Cell fating differentiates neurons. We only concern ourselves with neurons since we can ignore biological hardware.
# We need to represent not only the developing cells, but also the environment they develop within.

from typing import Callable, Self

from pydantic import BaseModel, Field

from sonin.model.neuron import TetanicPeriod
from sonin.model.signal import Signal, SignalCount
from sonin.model.stimulation import Stimulation

type Threshold = int
type IsLower = bool

# Represents a collection of predicates:
#     "does the count of this `Signal` meet the `Threshold` in the direction indicated by `IsLower`?"
type IsLeft = dict[tuple[Signal, IsLower], Threshold]


class FateNode(BaseModel):
    def get_fate(self, signals: dict[Signal, SignalCount]) -> "Fate":
        raise NotImplementedError("FateNode.get_fate")

    def size(self) -> int:
        raise NotImplementedError("FateNode.size")


class Fate(FateNode):
    """ The final configuration of a cell """

    excites: bool
    axon_signals: dict[Signal, SignalCount]
    activation_level: int = Field(ge=1)
    refactory_period: int = Field(ge=0)
    stimulation: Stimulation
    tetanic_period: TetanicPeriod | None

    def get_fate(self, signals: dict[Signal, SignalCount]) -> Self:
        return self

    def size(self) -> int:
        return 1


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

    def get_fate(self, signals: dict[Signal, SignalCount]) -> Fate:
        if all(
            signals.get(signal, 0) <= threshold if is_lower else signals.get(signal, 0) >= threshold
            for (signal, is_lower), threshold in self.is_left.items()
        ):
            return self.left.get_fate(signals)
        else:
            return self.right.get_fate(signals)

    def size(self) -> int:
        return self.left.size() + self.right.size()


class FateTree(BaseModel):
    root: FateNode | None = None

    def get_fate(self, signals: dict[Signal, SignalCount]) -> Fate | None:
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
