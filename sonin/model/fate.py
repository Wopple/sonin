# Cell fating differentiates neurons. We only concern ourselves with neurons since we can ignore biological hardware.
# We need to represent not only the developing cells, but also the environment they develop within.

from typing import Callable, Self

from pydantic import BaseModel, Field

from sonin.model.neuron import TetanicPeriod
from sonin.model.signal import Signal, SignalCount
from sonin.model.stimulation import Stimulation
from sonin.tree import BinaryTree

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


class FateTree(BinaryTree):
    root: FateNode | None = None
    is_leaf: Callable[[FateNode], bool] = lambda n: isinstance(n, Fate)

    def get_fate(self, signals: dict[Signal, SignalCount]) -> Fate | None:
        return self.root.get_fate(signals) if self.root is not None else None
