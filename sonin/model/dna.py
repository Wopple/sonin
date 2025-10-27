from sonin.model.facilitation import Facilitation
from sonin.model.mutation import FacilitationMutagen, Mutator, UintMutagen


class Dna:
    def __init__(
        self,
        min_neurons: int = 1,
        n_synapse: int = 1,
        n_dimension: int = 1,
        activation_level: int = 1,
        max_neuron_strength: int = 1,
        axon_range: int = 1,
        refactory_period: int = 0,
        facilitation_granularity: int = 1,
        facilitation_limit: int = 1,
    ):
        # Lower limit on the number of neurons in the hypercube
        self._min_neurons: UintMutagen = UintMutagen(int_value=min_neurons, min_value=1)

        # Number of synapses per neuron
        self._n_synapse: UintMutagen = UintMutagen(int_value=n_synapse, min_value=1)

        # Number of virtual spacial dimensions in a mind
        self._n_dimension: UintMutagen = UintMutagen(int_value=n_dimension, min_value=1, max_value=5)

        # Level at which a neuron activates
        self._activation_level: UintMutagen = UintMutagen(int_value=activation_level, min_value=1)

        # Neurons cannot propagate more potential than this when activating
        self._max_neuron_strength: UintMutagen = UintMutagen(int_value=max_neuron_strength, min_value=1)

        # Maximum city block distance an axon can reach away from the neuron
        self._axon_range: UintMutagen = UintMutagen(int_value=axon_range, min_value=1)

        # Amount of time a neuron stays in the refactory state (min 1)
        self._refactory_period: UintMutagen = UintMutagen(int_value=refactory_period)

        self._facilitation: FacilitationMutagen = FacilitationMutagen(
            granularity=UintMutagen(int_value=facilitation_granularity, min_value=1),
            limit=UintMutagen(int_value=facilitation_limit, min_value=1),
        )

        self.mutator: Mutator = Mutator(mutagens=[
            self._min_neurons,
            self._n_synapse,
            self._n_dimension,
            self._activation_level,
            self._max_neuron_strength,
            self._refactory_period,
            self._facilitation,
        ])

        self.initialize()

    def initialize(self):
        # Size of the hypercube of neurons, rounded up from the minimum number of neurons such that
        # dimension_size ^ n_dimension >= min_neurons

        dimension_size: int = 1

        while dimension_size ** self.n_dimension < self.min_neurons:
            dimension_size += 1

        self.dimension_size: int = dimension_size

        # Total number of neurons
        self.n_neuron: int = self.dimension_size ** self.n_dimension

    @property
    def min_neurons(self) -> int:
        return self._min_neurons.value

    @property
    def n_synapse(self) -> int:
        return self._n_synapse.value

    @property
    def n_dimension(self) -> int:
        return self._n_dimension.value

    @property
    def activation_level(self) -> int:
        return self._activation_level.value

    @property
    def max_neuron_strength(self) -> int:
        return self._max_neuron_strength.value

    @property
    def axon_range(self) -> int:
        return self._axon_range.value

    @property
    def refactory_period(self) -> int:
        return self._refactory_period.value

    @property
    def facilitation(self) -> Facilitation:
        return self._facilitation.value

    def mutate(self, num_mutations: int):
        self.mutator.mutate(num_mutations)
        self.initialize()
