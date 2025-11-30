from typing import Any, Self

from pydantic import BaseModel

from sonin.model.fate import Fate
from sonin.model.paint import CompleteFill, FillShape, RectangleShape, Shape


class Dna(BaseModel):
    num_dimensions: int
    dimension_size: int
    max_synapses: int
    max_synapse_strength: int
    max_axon_range: int

    # the first paint must always be a FillShape-CompleteFill paint to serve as a default
    fate_paints: list[tuple[Shape, Fate]]

    input_shape: Shape = None
    output_shape: Shape = None
    reward_shape: Shape | None = None
    punish_shape: Shape | None = None

    def model_post_init(self, context: Any, /):
        assert isinstance(self.fate_paints[0][0], FillShape)
        assert isinstance(self.fate_paints[0][0].fill, CompleteFill)

        if self.input_shape is None:
            self.input_shape = RectangleShape(sizes=(3,) * self.num_dimensions)

        if self.output_shape is None:
            self.output_shape = RectangleShape(sizes=(3,) * self.num_dimensions)

    @property
    def num_neurons(self) -> int:
        return self.dimension_size ** self.num_dimensions

    @classmethod
    def from_defaults(cls, num_dimensions: int = 2, dimension_size: int = 6) -> Self:
        return Dna(
            num_dimensions=num_dimensions,
            dimension_size=dimension_size,
            max_synapses=1,
            max_synapse_strength=1,
            max_axon_range=1,
            fate_paints=[(FillShape(fill=CompleteFill()), Fate.from_defaults(num_dimensions))],
        )
