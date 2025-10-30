from typing import Any, Callable, Generator, Iterator, Self

from pydantic import BaseModel, Field

from sonin.sonin_math import div


class Vector(BaseModel):
    # Virtual path of indices to a neuron in the hypercube
    value: tuple[int, ...]

    # Size of each dimension
    dimension_size: int | None = None

    # Index of the position in the single dimensional representation of the hypercube
    # Requires dimension_size to calculate
    index: int | None = None

    @classmethod
    def of(cls, value: tuple[int, ...] | list[int], dimension_size: int | None = None):
        return Vector(
            value=tuple(value),
            dimension_size=dimension_size,
        )

    def model_post_init(self, context: Any, /):
        if self.dimension_size is not None:
            self.index = sum(v * self.dimension_size ** i for i, v in enumerate(reversed(self.value)))

    @property
    def n_dimension(self) -> int:
        return len(self.value)

    def __eq__(self, other: Self):
        return self.index == other.index

    def __ne__(self, other: Self):
        return not self.__eq__(other)

    def __lt__(self, other: Self):
        return self.value < other.value

    def __le__(self, other: Self):
        return self.value <= other.value

    def __gt__(self, other: Self):
        return self.value > other.value

    def __ge__(self, other: Self):
        return self.value >= other.value

    def __hash__(self) -> int:
        return self.index

    def __neg__(self):
        return Vector.of(tuple(-v for v in self.value), self.dimension_size)

    def __add__(self, other: int | tuple[int, ...] | list[int] | Self) -> Self:
        if isinstance(other, int):
            return Vector.of(tuple(v + other for v in self.value), self.dimension_size)
        elif isinstance(other, tuple | list):
            assert self.n_dimension == len(other), f"{self.n_dimension} != {len(other)}"

            return Vector.of(tuple(s + o for s, o in zip(self.value, other)), self.dimension_size)
        elif isinstance(other, Vector):
            assert self.dimension_size == other.dimension_size, f"{self.dimension_size} != {other.dimension_size}"
            assert self.n_dimension == other.n_dimension, f"{self.n_dimension} != {other.n_dimension}"

            return Vector.of(tuple(s + o for s, o in zip(self.value, other.value)), self.dimension_size)
        else:
            raise TypeError(f"Vector.__add__ unexpected type: {other}")

    def __radd__(self, other: int) -> Self:
        return self + other

    def __sub__(self, other: int | Self) -> Self:
        if isinstance(other, int):
            return self + -other
        elif isinstance(other, tuple | list):
            return self + [-o for o in other]
        elif isinstance(other, Vector):
            return self + -other
        else:
            raise TypeError(f"Vector.__mul__ unexpected type: {other}")

    def __rsub__(self, other: int) -> Self:
        return -self + other

    def __mul__(self, other: int | Self) -> int | Self:
        if isinstance(other, int):
            return Vector.of(tuple(v * other for v in self.value), self.dimension_size)
        elif isinstance(other, Vector):
            assert self.dimension_size == other.dimension_size, f"{self.dimension_size} != {other.dimension_size}"

            return sum(self.value[idx] * other.value[idx] for idx in range(self.n_dimension))
        else:
            raise TypeError(f"Vector.__mul__ unexpected type: {other}")

    def __rmul__(self, other: int | Self) -> Self:
        return self * other

    def __truediv__(self, other: int | Self) -> Self:
        return self // other

    def __rtruediv__(self, other: int | Self) -> Self:
        return other // self

    def __floordiv__(self, other: int | Self) -> Self:
        if isinstance(other, int):
            return Vector.of(tuple(div(v, other) for v in self.value), self.dimension_size)
        elif isinstance(other, Vector):
            return Vector.of(tuple(div(vs, vo) for vs, vo in zip(self.value, other.value)), self.dimension_size)
        else:
            raise TypeError(f"Vector.__floordiv__ unexpected type: {other}")

    def __rfloordiv__(self, other: int | Self) -> Self:
        if isinstance(other, int):
            return Vector.of(tuple(div(other, v) for v in self.value), self.dimension_size)
        elif isinstance(other, Vector):
            return Vector.of(tuple(div(vo, vs) for vs, vo in zip(self.value, other.value)), self.dimension_size)
        else:
            raise TypeError(f"Vector.__rfloordiv__ unexpected type: {other}")

    def out_of_bounds(self) -> bool:
        return any(v < 0 or v >= self.dimension_size for v in self.value)

    def clip(self) -> Self:
        def clip(value: int, lower: int, upper: int) -> int:
            if value < lower:
                return lower
            elif value >= upper:
                return self.dimension_size - 1
            else:
                return value

        return Vector.of(tuple(clip(v, 0, self.dimension_size) for v in self.value), self.dimension_size)

    def grow(self, other: int) -> Self:
        """
        Grow the position by a single index
        """
        return Vector.of(self.value + (other,), self.dimension_size)

    def city_distance(self, other: tuple[int, ...] | list[int] | Self) -> int:
        """
        Integer based distance function using city block distance
        >>> Vector.of((1, 2)).city_distance([3, 1])
        3
        """
        if isinstance(other, Vector):
            other = other.value

        return sum(abs(a - b) for a, b in zip(self.value, other, strict=True))

    def city_unit(self) -> Self:
        """
        Approximate algorithm for finding the city unit vector that has the smallest
        angle with the current vector.
        """
        largest = max(abs(c) for c in self.value)

        if largest == 0:
            return Vector.of((0,) * self.n_dimension, self.dimension_size)

        # This algorithm will be close and usually correct, but not always.
        # This algorithm can be improved if necessary by checking the adjacent
        # positions or by doing a proper cosine similarity check.
        # The intuition is, if a component of the vector has a relatively high
        # magnitude compared to the other components, the unit vector likely
        # has a magnitude of 1 in the same direction.
        def approximate_coordinate(c: int) -> int:
            if abs(c) >= div(largest, 2):
                if c > 0:
                    return 1
                else:
                    return -1
            else:
                return 0

        return Vector.of(tuple(approximate_coordinate(c) for c in self.value), self.dimension_size)


class Shape(BaseModel):
    center: Vector
    size: int = Field(ge=1)

    def positions(self) -> Generator[Vector, None, None]:
        raise NotImplementedError(f"{self.__class__.__name__}.positions")


class CubeShape(Shape):
    """
    Produces shapes like:

    size: 1
    CT

    size: 2
    [][][]
    []CT[]
    [][][]

    size: 3
    [][][][][]
    [][][][][]
    [][]CT[][]
    [][][][][]
    [][][][][]
    """

    def positions(self) -> Generator[Vector, None, None]:
        # Iterate through layers. Recursively call for each lower dimension with the size of the layer.
        def iterate(
            dimension: int,
            position: tuple[int, ...],
        ) -> Generator[tuple[int, ...], None, None]:
            for idx in range(self.size * 2 - 1):
                relative_idx = idx - self.size + 1
                absolute_idx = relative_idx + self.center.value[dimension]

                if dimension == 0:
                    yield (absolute_idx,) + position
                else:
                    yield from iterate(dimension - 1, (absolute_idx,) + position)

        # only yield positions that are in bounds
        for value in iterate(self.center.n_dimension - 1, ()):
            candidate_position = Vector.of(value, self.center.dimension_size)

            if not candidate_position.out_of_bounds():
                yield candidate_position


class CityShape(Shape):
    """
    Produces shapes like:

    size: 1
    CT

    size: 2
      []
    []CT[]
      []

    size: 3
        []
      [][][]
    [][]CT[][]
      [][][]
        []
    """

    def positions(self) -> Generator[Vector, None, None]:
        # Iterate through layers. Recursively call for each lower dimension with the size of the layer.
        def iterate(
            dimension: int,
            size: int,
            position: tuple[int, ...],
        ) -> Generator[tuple[int, ...], None, None]:
            for idx in range(size * 2 - 1):
                relative_idx = idx - size + 1
                absolute_idx = relative_idx + self.center.value[dimension]

                if dimension == 0:
                    yield (absolute_idx,) + position
                else:
                    yield from iterate(dimension - 1, size - abs(relative_idx), (absolute_idx,) + position)

        # only yield positions that are in bounds
        for value in iterate(self.center.n_dimension - 1, self.size, ()):
            candidate_position = Vector.of(value, self.center.dimension_size)

            if not candidate_position.out_of_bounds():
                yield candidate_position


class Hypercube[T](BaseModel):
    n_dimension: int
    dimension_size: int
    items: list[T] = Field(default_factory=list)

    def __iter__(self) -> Iterator[T]:
        return iter(self.items)

    def initialize(self, create_item: Callable[[Vector], T]):
        def create_items(n_dimension: int, position: Vector) -> Generator[T, None, None]:
            if n_dimension == 0:
                # Only yield completed vectors.
                yield create_item(position)
            else:
                # Create a copy for every possible next index and recurse on each of them
                # yielding all results.
                for p in range(self.dimension_size):
                    yield from create_items(n_dimension - 1, position.grow(p))

        self.items = list(create_items(self.n_dimension, Vector.of((), self.dimension_size)))

    def get(self, position: int | tuple[int, ...] | list[int] | Vector) -> T:
        if isinstance(position, int):
            return self.items[position]
        elif isinstance(position, tuple | list):
            return self.items[Vector.of(tuple(position), self.dimension_size).index]
        else:
            return self.items[position.index]

    def center(self) -> list[T]:
        single = self.dimension_size % 2 == 1

        # for even dimension sizes, this is the corner with the smallest indices
        center = Vector.of(
            (div(self.dimension_size - 1, 2),) * self.n_dimension,
            self.dimension_size,
        )

        if single:
            # odd dimension sizes have a single center element
            return [self.get(center)]
        else:
            # even dimension sizes have a cluster of center elements
            def permute(dimension: int, edit: list[int]) -> Generator[list[int], None, None]:
                if dimension == 0:
                    yield edit
                else:
                    yield from permute(dimension - 1, edit + [0])
                    yield from permute(dimension - 1, edit + [1])

            return [
                self.get(center + edit)
                for edit in permute(self.n_dimension, [])
            ]
