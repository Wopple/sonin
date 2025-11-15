# Cell fating differentiates neurons.

from typing import Self

from pydantic import BaseModel, Field

from sonin.model.neuron import TetanicPeriod
from sonin.model.stimulation import Stimulation


class Fate(BaseModel):
    """ The final configuration of a cell """

    excites: bool
    axon_offset: tuple[int, ...]
    activation_level: int = Field(ge=1)
    refactory_period: int = Field(ge=0)
    stimulation: Stimulation
    overstimulation_threshold: int = Field(ge=1)
    tetanic_period: TetanicPeriod

    @classmethod
    def from_defaults(cls, num_dimensions: int) -> Self:
        return Fate(
            excites=True,
            axon_offset=(0,) * num_dimensions,
            activation_level=1,
            refactory_period=0,
            stimulation=Stimulation(),
            overstimulation_threshold=1,
            tetanic_period=TetanicPeriod(),
        )
