from dataclasses import dataclass, field
from typing import Callable, Generator, Self


@dataclass
class Vector:
    # Size of each dimension
    dimension_size: int

    # Virtual path of indices to a neuron in the hypercube
    value: tuple[int, ...]

    # Index of the position in the single dimensional representation of the hypercube
    index: int = field(init=False)

    def __post_init__(self):
        self.index = sum(v * self.dimension_size ** i for i, v in enumerate(reversed(self.value)))

    def grow(self, other: int) -> Self:
        """
        Grow the position by a single index
        """
        return Vector(self.dimension_size, self.value + (other,))

    def city_distance(self, other: Self) -> int:
        """
        Integer based distance function using city block distance
        >>> Vector(4, (1, 2)).city_distance(Vector(4, (3, 0)))
        4
        """
        return sum(abs(a - b) for a, b in zip(self.value, other.value, strict=True))

    def city_unit(self) -> Self:
        """
        Approximate algorithm for finding the city unit position that has the smallest
        angle with the current position.
        """
        largest = max(abs(c) for c in self.value)

        if largest == 0:
            return Vector(
                self.dimension_size,
                tuple(0 for _ in range(len(self.value))),
            )

        # This algorithm will be close and usually correct, but not always.
        # This algorithm can be improved if necessary by checking the adjacent
        # positions or by doing a proper cosine similarity check.
        def approximate_coordinate(c: int) -> int:
            if largest - abs(c) <= largest // 2:
                if c > 0:
                    return 1
                else:
                    return -1
            else:
                return 0

        return Vector(self.dimension_size, tuple(approximate_coordinate(c) for c in self.value))


@dataclass
class Hypercube[T]:
    n_dimension: int
    dimension_size: int
    items: list[T] = field(default_factory=list)

    def __iter__(self):
        return iter(self.items)

    def initialize(self, create_item: Callable[[Vector], T]):
        def create_items(n_dimension: int, position: Vector) -> Generator[T, None, None]:
            if n_dimension == 0:
                yield create_item(position)
            else:
                for p in range(self.dimension_size):
                    yield from create_items(n_dimension - 1, position.grow(p))

        self.items = list(create_items(self.n_dimension, Vector(self.dimension_size, ())))

    def get(self, position: int | Vector) -> T:
        if isinstance(position, int):
            return self.items[position]
        else:
            return self.items[position.index]
