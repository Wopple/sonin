from snn.model.dna import Dna
from snn.model.hypercube import Position

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

        # True if the neuron excites other neurons, False if it inhibits other neurons.
        self.excites: bool = excites

        # Positions of the neurons this neuron is currently connected to
        self.synapses: list[Position] = []

        # Current activation potential of the neuron
        self.potential: int = 0

        # Current state of the neuron
        self.state: str = ACCEPTING

        # Time at which to reactivate the neuron
        self.t_refactory_end: int = 0

        self.initialize()

    @property
    def inhibits(self) -> bool:
        return not self.excites

    def initialize(self):
        self.synapses = [None] * self.dna.n_synapse
        self.potential = 0
        self.state = ACCEPTING

    def enable(self):
        self.state = ACCEPTING

    def activate(self):
        self.potential = 0
        self.state = ACTIVATED

    def refactor(self, c_time: int):
        self.state = REFACTORY
        self.t_refactory_end = c_time + self.dna.refactory_period
