from math import ceil, log


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
