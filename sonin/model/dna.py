class Dna:
    def __init__(
        self,
        min_neurons: int = 1,
        n_synapse: int = 1,
        n_dimension: int = 1,
        activation_level: int = 1,
        max_neuron_strength: int = 2,
        refactory_period: int = 0,
        facilitation_granularity: int = 1,
        facilitation_limit: int = 1,
    ):
        # Number of synapses per neuron
        self.n_synapse: int = n_synapse

        # Number of virtual spacial dimensions in a mind
        self.n_dimension: int = n_dimension

        # Level at which a neuron activates
        self.activation_level: int = activation_level

        # Neurons cannot propagate more potential than this when activating
        self.max_neuron_strength: int = max_neuron_strength

        # Amount of time a neuron stays in the refactory state (min 1)
        self.refactory_period: int = refactory_period

        self.facilitation_granularity: int = facilitation_granularity
        self.facilitation_limit: int = facilitation_limit

        # Size of the hypercube of neurons, rounded up from the minimum number of neurons such that
        # dimension_size ^ n_dimension >= min_neurons

        dimension_size: int = 1

        while dimension_size ** n_dimension < min_neurons:
            dimension_size += 1

        self.dimension_size: int = dimension_size

        # Total number of neurons
        self.n_neuron: int = self.dimension_size ** n_dimension
