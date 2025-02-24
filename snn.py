from math import ceil, log
from random import seed, randint
from typing import Self, Generator

# Regulate excitation when overstimulated by weakening connections

# Will accept potential from pre-synaptic neurons
ACCEPTING = 'accepting'

# Will send potential to post-synaptic neurons
ACTIVATED = 'activated'

# Will reject potential from pre-synaptic neurons
REFACTORY = 'refactory'


class Dna:
    def __init__(
        self,
        n_synapse: int,
        n_dimension: int,
        activation_level: int,
        refactory_period: int,
        min_neurons: int,
    ):
        # Number of synapses per neuron
        self.n_synapse: int = n_synapse

        # Number of virtual spacial dimensions in a mind
        self.n_dimension: int = n_dimension

        # Level at which a neuron activates
        self.activation_level: int = activation_level

        # Amount of time a neuron stays in the refactory state (min 1)
        self.refactory_period: int = refactory_period

        # Size of each list in a mind's neurons, rounded up from the minimum number of neurons such that
        # dimension_size ^ n_dimension >= min_neurons
        self.dimension_size: int = int(ceil(log(min_neurons, n_dimension)))

        # Total number of neurons
        self.n_neuron: int = self.dimension_size ** n_dimension


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


def random_position(dna: Dna) -> Position:
    return Position(dna.dimension_size, tuple(randint(0, dna.dimension_size - 1) for _ in range(dna.n_dimension)))


class Neuron:
    def __init__(self, dna: Dna, position: Position):
        self.dna: Dna = dna

        # Position of the neuron in the mind
        self.position: Position = position

        # Positions of the neurons this neuron is currently connected to
        self.synapses: list[Position] = []

        # Current activation potential of the neuron
        self.potential: int = 0

        # Current state of the neuron
        self.state: str = ACCEPTING

        # Time at which to reactivate the neuron
        self.t_refactory_end: int = 0

        self.initialize()

    def initialize(self):
        self.synapses = [None] * dna.n_synapse
        self.potential = 0
        self.state = ACCEPTING

    def activate(self):
        self.potential = 0
        self.state = ACTIVATED

    def refactor(self, c_time: int):
        self.state = REFACTORY
        self. t_refactory_end = c_time + self.dna.refactory_period


class Hypercube:
    def __init__(self, dna: Dna):
        self.dna: Dna = dna
        self.items: list[Neuron] = []
        self.initialize()

    def __iter__(self):
        return iter(self.items)

    def initialize(self):
        def create_neurons(n_dimension: int, position: Position) -> Generator[Neuron, None, None]:
            if n_dimension == 0:
                yield Neuron(self.dna, position)
            else:
                for p in range(self.dna.dimension_size):
                    yield from create_neurons(n_dimension - 1, position.grow(p))

        self.items = list(create_neurons(self.dna.n_dimension, Position(self.dna.dimension_size, ())))

    def get(self, position: int | Position) -> Neuron:
        if isinstance(position, int):
            return self.items[position]
        else:
            return self.items[position.index]


class Mind:
    def __init__(self, dna: Dna):
        self.dna: Dna = dna
        self.neurons: Hypercube = Hypercube(dna)

    def randomize_synapses(self):
        for n in self.neurons:
            for i in range(self.dna.n_synapse):
                n.synapses[i] = random_position(self.dna)


seed(0)

dna = Dna(
    min_neurons=100,
    n_synapse=4,
    n_dimension=2,
    activation_level=2,
    refactory_period=2,
)

mind = Mind(dna)
mind.randomize_synapses()

for n in mind.neurons:
    print(n.position.value, [s.value for s in n.synapses])
