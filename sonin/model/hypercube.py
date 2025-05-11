from dataclasses import dataclass, field
from typing import Callable, Generator, Self

from sonin.model.math import div


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

    @property
    def n_dimension(self) -> int:
        return len(self.value)

    def __eq__(self, other: Self):
        return self.index == other.index

    def __ne__(self, other: Self):
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return self.index

    def __add__(self, other: int | Self) -> Self:
        if isinstance(other, int):
            return Vector(
                self.dimension_size,
                tuple(v + other for v in self.value),
            )
        elif isinstance(other, Vector):
            assert self.dimension_size == other.dimension_size, f"{self.dimension_size} != {other.dimension_size}"

            return Vector(
                self.dimension_size,
                tuple(self.value[idx] + other.value[idx] for idx in range(self.n_dimension)),
            )
        else:
            raise TypeError(f"Vector.__add__: unexpected type: {other}")

    def __radd__(self, other: int) -> Self:
        return self + other

    def __sub__(self, other: int | Self) -> Self:
        if isinstance(other, int):
            return Vector(
                self.dimension_size,
                tuple(v - other for v in self.value),
            )
        elif isinstance(other, Vector):
            assert self.dimension_size == other.dimension_size, f"{self.dimension_size} != {other.dimension_size}"

            return Vector(
                self.dimension_size,
                tuple(self.value[idx] - other.value[idx] for idx in range(self.n_dimension)),
            )
        else:
            raise TypeError(f"Vector.__mul__: unexpected type: {other}")

    def __rsub__(self, other: int) -> Self:
        return self - other

    def __mul__(self, other: int | Self) -> int | Self:
        if isinstance(other, int):
            return Vector(
                self.dimension_size,
                tuple(v * other for v in self.value),
            )
        elif isinstance(other, Vector):
            assert self.dimension_size == other.dimension_size, f"{self.dimension_size} != {other.dimension_size}"

            return sum(self.value[idx] * other.value[idx] for idx in range(self.n_dimension))
        else:
            raise TypeError(f"Vector.__mul__: unexpected type: {other}")

    def __rmul__(self, other: int) -> Self:
        return self * other

    def __truediv__(self, other: int) -> Self:
        return Vector(
            self.dimension_size,
            tuple(div(v, other) for v in self.value),
        )

    def __rtruediv__(self, other: int) -> Self:
        return self / other

    def clip(self) -> Self:
        def clip(value: int, lower: int, upper: int) -> int:
            if value < lower:
                return lower
            elif value >= upper:
                return self.dimension_size - 1
            else:
                return value

        return Vector(
            self.dimension_size,
            tuple(clip(v, 0, self.dimension_size) for v in self.value),
        )

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
        Approximate algorithm for finding the city unit vector that has the smallest
        angle with the current vector.
        """
        largest = max(abs(c) for c in self.value)

        if largest == 0:
            return Vector(
                self.dimension_size,
                tuple(0 for _ in range(self.n_dimension)),
            )

        # This algorithm will be close and usually correct, but not always.
        # This algorithm can be improved if necessary by checking the adjacent
        # positions or by doing a proper cosine similarity check.
        def approximate_coordinate(c: int) -> int:
            if abs(c) >= div(largest, 2):
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
