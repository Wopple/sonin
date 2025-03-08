from typing import Callable, Generator, Self

from snn.model.dna import Dna


class Position:
    def __init__(self, dimension_size: int, value: tuple[int, ...]):
        self.dimension_size: int = dimension_size

        # Virtual path of indices to a neuron in the hypercube
        self.value: tuple[int, ...] = value

        # Index of the position in the single dimensional representation of the hypercube
        self.index: int = sum(v * dimension_size ** i for i, v in enumerate(reversed(value)))

    def grow(self, other: int) -> Self:
        """
        Grow the position by a single index
        """
        return Position(self.dimension_size, self.value + (other,))

    def city_distance(self, other: Self) -> int:
        """
        Integer based distance function
        >>> Position(4, (1, 2)).city_distance(Position(4, (3, 0)))
        4
        """
        return sum(abs(a - b) for a, b in zip(self.value, other.value, strict=True))


class Hypercube[T]:
    def __init__(self, dna: Dna):
        self.n_dimension: int = dna.n_dimension
        self.dimension_size: int = dna.dimension_size
        self.items: list[T] = []

    def __iter__(self):
        return iter(self.items)

    def initialize(self, create_item: Callable[[Position], T]):
        def create_items(n_dimension: int, position: Position) -> Generator[T, None, None]:
            if n_dimension == 0:
                yield create_item(position)
            else:
                for p in range(self.dimension_size):
                    yield from create_items(n_dimension - 1, position.grow(p))

        self.items = list(create_items(self.n_dimension, Position(self.dimension_size, ())))

    def get(self, position: int | Position) -> T:
        if isinstance(position, int):
            return self.items[position]
        else:
            return self.items[position.index]
