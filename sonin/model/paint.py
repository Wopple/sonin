from typing import Generator, Literal, Self

from pydantic import BaseModel, Field, model_validator

from sonin.model.hypercube import Position, Vector
from sonin.sonin_math import div

type Fill = CompleteFill | ModuloFill | OffsetFill

class BaseFill(BaseModel):
    def contains(self, vector: Vector) -> bool:
        raise NotImplementedError(f'{self.__class__.__name__}.contains')


class CompleteFill(BaseFill):
    ty: Literal['Complete'] = 'Complete'

    def contains(self, vector: Vector) -> bool:
        return True


class ModuloFill(BaseFill):
    ty: Literal['Modulo'] = 'Modulo'
    divisor: int = Field(ge=2)
    remainder: int = Field(ge=0)

    @model_validator(mode='after')
    def validate_model(self) -> Self:
        assert self.remainder < self.divisor

        return self

    def contains(self, vector: Vector) -> bool:
        return sum(vector.value) % self.divisor == self.remainder


class OffsetFill(BaseFill):
    ty: Literal['Offset'] = 'Offset'
    base: tuple[int, ...]
    offsets: tuple[int, ...]

    cache_key: tuple[int, tuple[int, ...], tuple[int, ...]] | None = Field(default=None, exclude=True)
    cache_values: set[tuple[int, ...]] = Field(default=set(), exclude=True)

    def contains(self, vector: Vector) -> bool:
        cache_key = (vector.dimension_size, self.base, self.offsets)

        if self.cache_key != cache_key:
            assert all(v < vector.dimension_size for v in self.base)
            assert all(offset < vector.dimension_size for offset in self.offsets)

            self.cache_key = cache_key
            self.cache_values = {self.base}
            current = [v for v in self.base]

            while True:
                for idx, offset in enumerate(self.offsets):
                    current[idx] = (current[idx] + offset) % vector.dimension_size

                next_tuple = tuple(current)

                if next_tuple in self.cache_values:
                    break
                else:
                    self.cache_values.add(tuple(current))

        return vector.value in self.cache_values


type Shape = FillShape | RectangleShape | CityShape


class BaseShape(BaseModel):
    def positions(self, num_dimensions: int, dimension_size: int) -> Generator[Vector, None, None]:
        raise NotImplementedError(f'{self.__class__.__name__}.positions')


class FillShape(BaseShape):
    ty: Literal['Fill'] = 'Fill'

    # config
    fill: Fill = Field(default_factory=CompleteFill)
    outline: bool = False

    def positions(self, num_dimensions: int, dimension_size: int) -> Generator[Vector, None, None]:
        def iterate(dim: int = 0, value: tuple[int, ...] = ()) -> Generator[tuple[int, ...], None, None]:
            if dim == num_dimensions:
                yield value
            else:
                if self.outline and 0 not in value and (dimension_size - 1) not in value:
                    yield from iterate(dim + 1, value + (0,))
                    yield from iterate(dim + 1, value + (dimension_size - 1,))
                else:
                    for i in range(dimension_size):
                        yield from iterate(dim + 1, value + (i,))

        for value in iterate():
            position = Vector.of(value, dimension_size)

            if self.fill.contains(position):
                yield position


class RectangleShape(BaseShape):
    """
    Produces shapes like:

    sizes: (1, 1)
    CT

    sizes: (2, 1)
    CT[]

    sizes: (2, 3)
    [][]
    CT[]
    [][]

    sizes: (5, 3)
    [][][][][]
    [][]CT[][]
    [][][][][]
    """

    ty: Literal['Rectangle'] = 'Rectangle'

    # shape definition
    center: Position | None = None
    sizes: tuple[int, ...]

    # config
    fill: Fill = Field(default_factory=CompleteFill)
    outline: bool = False
    wrap: bool = False

    def positions(self, num_dimensions: int, dimension_size: int) -> Generator[Vector, None, None]:
        assert all(size <= dimension_size for size in self.sizes)

        center: Vector = self.center.get(dimension_size)

        lowers = [center[dim] - div(self.sizes[dim] - 1, 2) for dim in range(num_dimensions)]
        uppers = [center[dim] + div(self.sizes[dim], 2) + 1 for dim in range(num_dimensions)]

        if not self.wrap:
            for idx in range(num_dimensions):
                lowers[idx] = max(0, lowers[idx])
                uppers[idx] = min(dimension_size, uppers[idx])

        def iterate(dim: int = 0, value: tuple[int, ...] = ()) -> Generator[tuple[int, ...], None, None]:
            if dim == num_dimensions:
                yield value
            else:
                for i in range(lowers[dim], uppers[dim]):
                    yield from iterate(dim + 1, value + (i,))

        for value in iterate():
            position = Vector.of(value, dimension_size)

            is_outline_value = all(
                value[dim] not in bounds
                for dim, bounds in enumerate(zip(lowers, (u - 1 for u in uppers)))
            )

            # skip values not in the outline
            if self.outline and is_outline_value:
                # if in wrap mode, city bounds check is enough
                # otherwise, also check not in the clipped outline
                if self.wrap or (0 not in value and (dimension_size - 1) not in value):
                    continue

            # wrap out-of-bounds values when in wrap mode
            # skip out-of-bounds values when not in wrap mode
            if position.out_of_bounds():
                if self.wrap:
                    position = Vector.of(tuple((v + dimension_size) % dimension_size for v in value), dimension_size)
                else:
                    continue

            if self.fill.contains(position):
                yield position


class CityShape(BaseShape):
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

    ty: Literal['City'] = 'City'

    # shape definition
    center: Position
    size: int = Field(ge=1)

    # config
    fill: Fill = Field(default_factory=CompleteFill)
    outline: bool = False
    wrap: bool = False

    def positions(self, num_dimensions: int, dimension_size: int) -> Generator[Vector, None, None]:
        assert self.size <= div(dimension_size + 1, 2)

        center: Vector = self.center.get(dimension_size)

        # Iterate through layers. Recursively call for each lower dimension with the size of the layer.
        def iterate(dim: int = 0, value: tuple[int, ...] = (), *, size: int) -> Generator[tuple[int, ...], None, None]:
            if dim == num_dimensions:
                yield value
            else:
                for relative_idx in range(1 - size, size):
                    absolute_idx = relative_idx + center[dim]

                    yield from iterate(
                        dim=dim + 1,
                        value=value + (absolute_idx,),
                        size=size - abs(relative_idx),
                    )

        for value in iterate(size=self.size):
            position = Vector.of(value, dimension_size)

            # skip values not in the outline
            if self.outline and center.city_distance(position) != self.size - 1:
                # if in wrap mode, city distance check is enough
                # otherwise, also check not in the clipped outline
                if self.wrap or (0 not in value and (dimension_size - 1) not in value):
                    continue

            # wrap out-of-bounds values when in wrap mode
            # skip out-of-bounds values when not in wrap mode
            if position.out_of_bounds():
                if self.wrap:
                    position = Vector.of(tuple((v + dimension_size) % dimension_size for v in value), dimension_size)
                else:
                    continue

            if self.fill.contains(position):
                yield position
