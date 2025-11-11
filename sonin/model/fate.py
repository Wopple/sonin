# Cell fating differentiates neurons. We only concern ourselves with neurons since we can ignore biological hardware.
# We need to represent not only the developing cells, but also the environment they develop within.

from typing import Any, Callable, Generator, Self

from pydantic import BaseModel, Field, field_serializer, field_validator

from sonin.model.neuron import TetanicPeriod
from sonin.model.signal import Signal, SignalCount
from sonin.model.stimulation import Stimulation
from sonin.tree import BinaryTree

type Threshold = int
type IsLower = bool

# Represents a collection of predicates:
#     "does the count of this `Signal` meet the `Threshold` in the direction indicated by `IsLower`?"
type IsLeft = list[tuple[Signal, IsLower, Threshold]]

KIND_FATE = 0
KIND_BINARY_FATE = 1


def parse_fate_node(node: Any) -> BaseModel:
    if isinstance(node, BaseModel):
        return node
    elif isinstance(node, dict):
        if node['kind'] == KIND_FATE:
            return Fate(**node)
        elif node['kind'] == KIND_BINARY_FATE:
            return BinaryFate(**node)

    raise ValueError(f'Unable to parse: {node}')


class FateNode(BaseModel):
    def get_fate(self, signals: dict[Signal, SignalCount]) -> 'Fate':
        raise NotImplementedError('FateNode.get_fate')

    def size(self) -> int:
        raise NotImplementedError('FateNode.size')


class Fate(FateNode):
    """ The final configuration of a cell """

    kind: int = KIND_FATE
    excites: bool
    axon_signals: dict[Signal, SignalCount]
    activation_level: int = Field(ge=1)
    refactory_period: int = Field(ge=0)
    stimulation: Stimulation
    overstimulation_threshold: int = Field(ge=1)
    tetanic_period: TetanicPeriod

    @classmethod
    def from_defaults(cls) -> Self:
        return Fate(
            excites=True,
            axon_signals={},
            activation_level=1,
            refactory_period=0,
            stimulation=Stimulation(),
            overstimulation_threshold=1,
            tetanic_period=TetanicPeriod(),
        )

    def get_fate(self, signals: dict[Signal, SignalCount]) -> Self:
        return self

    def size(self) -> int:
        return 1


class BinaryFate(FateNode):
    """ A branch in a decision tree """

    kind: int = KIND_BINARY_FATE
    left: FateNode
    right: FateNode
    is_left: IsLeft

    def __iter__(self) -> Generator[Fate, None, None]:
        if isinstance(self.left, Fate):
            yield self.left
        elif isinstance(self.left, BinaryFate):
            yield from self.left

        if isinstance(self.right, Fate):
            yield self.right
        elif isinstance(self.right, BinaryFate):
            yield from self.right

    def branches_iter(self) -> Generator[Self, None, None]:
        yield self

        if isinstance(self.left, BinaryFate):
            yield from self.left.branches_iter()

        if isinstance(self.right, BinaryFate):
            yield from self.right.branches_iter()

    @field_validator('left', 'right', mode='before')
    @classmethod
    def validate_fate_node(cls, node: dict[str, Any]) -> Any:
        return parse_fate_node(node)

    @field_serializer('left', 'right')
    @classmethod
    def serialize_fate_node(cls, node: BaseModel) -> str | dict[str, Any]:
        # necessary to call the specific serializer
        return node.model_dump()

    def get_fate(self, signals: dict[Signal, SignalCount]) -> Fate:
        if all(
            signals.get(signal, 0) <= threshold if is_lower else signals.get(signal, 0) >= threshold
            for signal, is_lower, threshold in self.is_left
        ):
            return self.left.get_fate(signals)
        else:
            return self.right.get_fate(signals)

    def size(self) -> int:
        return self.left.size() + self.right.size()


class FateTree(BinaryTree):
    root: FateNode | None = None
    is_leaf: Callable[[FateNode], bool] = Field(default=lambda n: isinstance(n, Fate), exclude=True)

    @field_validator('root', mode='before')
    @classmethod
    def validate_fate_node(cls, node: dict[str, Any]) -> Any:
        return parse_fate_node(node)

    @field_serializer('root')
    @classmethod
    def serialize_fate_node(cls, node: BaseModel) -> str | dict[str, Any]:
        # necessary to call the specific serializer
        return node.model_dump()

    def get_fate(self, signals: dict[Signal, SignalCount]) -> Fate | None:
        return self.root.get_fate(signals) if self.root is not None else None
