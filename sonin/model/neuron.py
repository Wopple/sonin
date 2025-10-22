from dataclasses import dataclass, field, InitVar
from typing import Self

from sonin.model.hypercube import Vector
from sonin.model.signal import Signal
from sonin.model.stimulation import SnapBack
from sonin.model.synapse import Synapse

# Will accept potential from pre-synaptic neurons
ACCEPTING = 0

# Will reject potential from pre-synaptic neurons
REFACTORY = 1


@dataclass
class TetanicPeriod:
    """ Represents a periodic schedule of tetanic activations """

    # Number of steps to begin tetanic activation
    threshold: int

    # Number of activations in each tetanus
    activations: int

    # Number of steps between activations during tetanus
    gap: int

    # True if dormant, False if within tetanus
    _dormant: bool = field(init=False)

    # Time of next flip between dormant and tetanus
    _n_time: int = field(init=False)

    def __post_init__(self):
        self._dormant: bool = True
        self._n_time: int = self.threshold

    def step(self, c_time: int):
        if self._dormant:
            if c_time >= self._n_time:
                self._dormant = False
                self._n_time = c_time + self.activations * (self.gap + 1)
        else:
            if c_time >= self._n_time:
                self._dormant = True
                self._n_time = c_time + self.threshold

    def is_active(self, c_time) -> bool:
        """ Returns True if the neuron should fire """
        return not self._dormant and (self._n_time - c_time) % (self.gap + 1) == 0


class NullTetanicPeriod(TetanicPeriod):
    def __new__(cls, *args, **kwargs) -> Self:
        if not hasattr(cls, "instance"):
            cls.instance: NullTetanicPeriod = super().__new__(cls)

        return cls.instance

    def __init__(self):
        pass

    def step(self, _c_time: int):
        pass

    def is_active(self, _c_time) -> bool:
        return False


@dataclass
class Axon:
    position: Vector
    n_dimension: InitVar[int]
    dimension_size: InitVar[int]
    signals: set[Signal] = field(default_factory=set)
    direction: Vector = field(init=False)

    def __post_init__(self, n_dimension: int, dimension_size: int):
        # All axons start out pointing at the center. This helps differentiate
        # neurons and expose them to the most signals.

        direction = ()

        for i in range(n_dimension):
            # Not halving the dimension_size because int division will not capture a center between points
            double = self.position.value[i] * 2

            if double < dimension_size:
                direction += (-1,)
            elif double > dimension_size:
                direction += (1,)
            else:
                direction += (0,)

        assert len(direction) == n_dimension

        self.direction = Vector(dimension_size, direction)


@dataclass
class Neuron:
    # Vector of the neuron in the mind
    position: Vector

    # Determines where synaptic connections can be made
    axon: Axon

    # The signals this neuron emits
    signals: set[Signal] = field(default_factory=set)

    # True if the neuron excites other neurons, False if it inhibits other neurons
    excites: bool = True

    # Current activation potential of the neuron
    potential: int = 0

    # Potential threshold for activation
    activation_level: int = 1

    # Number of dormant steps after activation
    refactory_period: int = 0

    # Periodic activations without need for input potential
    tetanic_period: TetanicPeriod = field(default_factory=lambda: NullTetanicPeriod())

    # Synapses connected to post neurons (output)
    post_synapses: dict[int, Synapse] = field(default_factory=dict)

    # Synapses connected to pre neurons (input)
    pre_synapses: dict[int, Synapse] = field(default_factory=dict)

    # Current state of the neuron
    state: int = ACCEPTING

    # If True, will send potential to post-synaptic neurons
    activated: bool = False

    # Time at which to reactivate the neuron
    t_refactory_end: int = 0

    stimulation: SnapBack = field(init=False)
    stimulation_amount: int = 64
    stimulation_restore_rate: InitVar[int] = 8
    stimulation_restore_damper: InitVar[int] = 7

    # Effective range for each signal emitted by this neuron
    effective_range: dict[Signal, int] = field(default_factory=dict)

    def __post_init__(
        self,
        stimulation_restore_rate: int,
        stimulation_restore_damper: int,
    ):
        # Detects frequent activations
        self.stimulation = SnapBack(
            restore_rate=stimulation_restore_rate,
            restore_damper=stimulation_restore_damper,
        )

    @property
    def inhibits(self) -> bool:
        return not self.excites

    def step(self, c_time: int):
        self.stimulation.step()
        self.tetanic_period.step(c_time)

        if self.state == ACCEPTING:
            if self.potential >= self.activation_level or self.tetanic_period.is_active(c_time):
                self.activate(c_time)
        elif self.state == REFACTORY and c_time >= self.t_refactory_end:
            self.enable()

    def enable(self):
        self.state = ACCEPTING

    def activate(self, c_time: int):
        self.stimulation.value += self.stimulation_amount
        self.potential = 0
        self.activated = True

        if self.refactory_period > 0:
            self.state = REFACTORY
            self.t_refactory_end = c_time + self.refactory_period

    def deactivate(self):
        self.activated = False
