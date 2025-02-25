from typing import Generator, Callable

from snn.model.dna import Dna
from snn.model.position import Position


class Hypercube[T]:
    def __init__(self, dna: Dna):
        self.n_dimension: int = dna.n_dimension
        self.dimension_size: int = dna.dimension_size
        self.items: list = []

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
