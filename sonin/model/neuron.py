from sonin.model.dna import Dna
from sonin.model.hypercube import Position
from sonin.model.stimulation import SnapBack
from sonin.model.synapse import Synapse

# Will accept potential from pre-synaptic neurons
ACCEPTING = 'accepting'

# Will send potential to post-synaptic neurons
ACTIVATED = 'activated'

# Will reject potential from pre-synaptic neurons
REFACTORY = 'refactory'


class Neuron:
    def __init__(self, dna: Dna, position: Position, excites: bool = True):
        self.dna: Dna = dna

        # Position of the neuron in the mind
        self.position: Position = position

        # True if the neuron excites other neurons, False if it inhibits other neurons
        self.excites: bool = excites

        # Positions of the neurons this neuron is currently connected to
        self.synapses: list[Synapse] = []

        # Current activation potential of the neuron
        self._potential: int = 0

        # Current state of the neuron
        self.state: str = ACCEPTING

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
        self.synapses = [None] * self.dna.n_synapse
        self.potential = 0
        self.state = ACCEPTING

    def step(self, c_time: int):
        self.stimulation.step()

        if self.state == ACCEPTING and self.potential >= self.dna.activation_level:
            self.activate()
        elif self.state == REFACTORY and c_time > self.t_refactory_end:
            self.enable()

    def enable(self):
        self.state = ACCEPTING

    def activate(self):
        self.stimulation.value += 64
        self.potential = 0
        self.state = ACTIVATED

    def refactor(self, c_time: int):
        if self.dna.refactory_period > 0:
            self.state = REFACTORY
            self.t_refactory_end = c_time + self.dna.refactory_period
        else:
            self.enable()
