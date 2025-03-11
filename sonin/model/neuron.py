from sonin.model.dna import Dna
from sonin.model.hypercube import Position
from sonin.model.stimulation import SnapBack
from sonin.model.synapse import Synapse

# Will accept potential from pre-synaptic neurons
ACCEPTING = 'accepting'

# Will reject potential from pre-synaptic neurons
REFACTORY = 'refactory'

class TetanicPeriod:
    def __init__(self, threshold: int, activations: int, gap: int):
        # Number of steps to begin tetanic activation
        self.threshold: int = threshold

        # Number of activations in each tetanus
        self.activations: int = activations

        # Number of steps between activations during tetanus
        self.gap: int = gap

        # True if dormant, False if within tetanus
        self.dormant: bool = True

        # Time of next flip between dormant and tetanus
        self.n_time: int = threshold

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
        return not self.dormant and (self.n_time - c_time) % (self.gap + 1) == 0

class NullTetanicPeriod(TetanicPeriod):
    def __init__(self):
        pass

    def step(self, _c_time: int):
        pass

    def is_active(self, _c_time) -> bool:
        return False

class Neuron:
    def __init__(
        self,
        dna: Dna,
        position: Position,
        excites: bool = True,
        tetanic_period: TetanicPeriod = NullTetanicPeriod(),
    ):
        self.dna: Dna = dna

        # Position of the neuron in the mind
        self.position: Position = position

        # True if the neuron excites other neurons, False if it inhibits other neurons
        self.excites: bool = excites

        self.tetanic_period: TetanicPeriod = tetanic_period

        # Synapses connected to post neurons (output)
        self.post_synapses: dict[int, Synapse] = {}

        # Synapses connected to pre neurons (input)
        self.pre_synapses: dict[int, Synapse] = {}

        # Current activation potential of the neuron
        self._potential: int = 0

        # Current state of the neuron
        self.state: str = ACCEPTING

        # If True, will send potential to post-synaptic neurons
        self.activated: bool = False

        # Time at which to reactivate the neuron
        self.t_refactory_end: int = 0

        # Detects frequent activations
        self.stimulation = SnapBack(restore_rate=8, restore_scalar=7)

        self.initialize()

    @property
    def inhibits(self) -> bool:
        return not self.excites

    @property
    def potential(self) -> int:
        return self._potential

    @potential.setter
    def potential(self, potential):
        self._potential = potential

    def initialize(self):
        self.potential = 0
        self.state = ACCEPTING

    def step(self, c_time: int):
        self.stimulation.step()
        self.tetanic_period.step(c_time)

        if self.state == ACCEPTING:
            if self.potential >= self.dna.activation_level or self.tetanic_period.is_active(c_time):
                self.activate(c_time)
        elif self.state == REFACTORY and c_time >= self.t_refactory_end:
            self.enable()

    def enable(self):
        self.state = ACCEPTING

    def activate(self, c_time: int):
        self.stimulation.value += 64
        self.potential = 0
        self.activated = True

        if self.dna.refactory_period > 0:
            self.state = REFACTORY
            self.t_refactory_end = c_time + self.dna.refactory_period

    def deactivate(self):
        self.activated = False
