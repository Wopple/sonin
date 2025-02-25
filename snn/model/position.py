from typing import Self


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
