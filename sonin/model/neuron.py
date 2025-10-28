from typing import Any

from pydantic import BaseModel, Field

from sonin.model.hypercube import Vector
from sonin.model.signal import Signal, SignalCount
from sonin.model.stimulation import SnapBack, Stimulation
from sonin.model.synapse import Synapse

# Will accept potential from pre-synaptic neurons
ACCEPTING = 0

# Will reject potential from pre-synaptic neurons
REFACTORY = 1


class TetanicPeriod(BaseModel):
    """ Represents a periodic schedule of tetanic activations """

    # Number of steps to begin tetanic activation
    threshold: int

    # Number of activations in each tetanus
    activations: int

    # Number of steps between activations during tetanus
    gap: int

    # True if dormant, False if within tetanus
    dormant: bool = True

    # Time of next flip between dormant and tetanus
    n_time: int = None

    def model_post_init(self, context: Any, /):
        self.n_time: int = self.threshold

    def step(self, c_time: int):
        if self.dormant:
            if c_time >= self.n_time:
                self.dormant = False
                self.n_time = c_time + self.activations * (self.gap + 1)
        else:
            if c_time >= self.n_time:
                self.dormant = True
                self.n_time = c_time + self.threshold

    def is_active(self, c_time) -> bool:
        """ Returns True if the neuron should fire """
        return not self.dormant and (self.n_time - c_time) % (self.gap + 1) == 0


class Axon(BaseModel):
    position: Vector
    n_dimension: int
    dimension_size: int
    signals: dict[Signal, SignalCount] = Field(default_factory=dict)
    direction: Vector = None

    def model_post_init(self, context: Any, /):
        # All axons start out pointing at the center. This helps differentiate
        # neurons and expose them to the most signals.
        double_center = Vector.of(tuple(self.dimension_size - 1 for _ in range(self.n_dimension)), self.dimension_size)
        double_position = self.position * 2
        self.direction = (double_center - double_position).city_unit()


class Neuron(BaseModel):
    # Vector of the neuron in the mind
    position: Vector

    # Determines where synaptic connections can be made
    axon: Axon

    # The signals this neuron emits
    signals: dict[Signal, SignalCount] = Field(default_factory=set)

    # True if the neuron excites other neurons, False if it inhibits other neurons
    excites: bool = True

    # Current activation potential of the neuron
    potential: int = 0

    # Potential threshold for activation
    activation_level: int = Field(default=1, ge=1)

    # Number of dormant steps after activation
    refactory_period: int = Field(default=0, ge=0)

    # Periodic activations without need for input potential
    tetanic_period: TetanicPeriod | None = None

    # Synapses connected to post neurons (output)
    post_synapses: dict[int, Synapse] = Field(default_factory=dict)

    # Synapses connected to pre neurons (input)
    pre_synapses: dict[int, Synapse] = Field(default_factory=dict)

    # Current state of the neuron
    state: int = ACCEPTING

    # If True, will send potential to post-synaptic neurons
    activated: bool = False

    # Time at which to reactivate the neuron
    t_refactory_end: int = 0

    stimulation: Stimulation

    # Effective range for each signal emitted by this neuron
    effective_range: dict[Signal, int] = Field(default_factory=dict)

    @property
    def inhibits(self) -> bool:
        return not self.excites

    def step(self, c_time: int):
        self.stimulation.step()
        self.tetanic_period and self.tetanic_period.step(c_time)

        # activate if potential is exceeded or there is a tetanic activation
        if self.state == ACCEPTING:
            is_tetanic = self.tetanic_period and self.tetanic_period.is_active(c_time)

            if self.potential >= self.activation_level or is_tetanic:
                self.activate(c_time)
        # re-enable after the refactory period ends
        elif self.state == REFACTORY and c_time >= self.t_refactory_end:
            self.enable()

    def enable(self):
        self.state = ACCEPTING

    def activate(self, c_time: int):
        self.stimulation.stimulate()
        self.potential = 0
        self.activated = True

        if self.refactory_period > 0:
            self.state = REFACTORY
            self.t_refactory_end = c_time + self.refactory_period

    def deactivate(self):
        self.activated = False
