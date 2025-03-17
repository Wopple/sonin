from random import randint, shuffle

class Mutagen:
    def __init__(self, occurrence_weight: int | None = None, deviation_weight: int | None = None):
        self.occurrence_weight: int = occurrence_weight or 1
        self.deviation_weight: int = deviation_weight or 1

        assert self.occurrence_weight >= 1
        assert self.deviation_weight >= 1

    def mutate(self, num: int):
        raise NotImplementedError("Mutagen.mutate")

class IntMutagen(Mutagen):
    MIN: int = -(2 ** 63)
    MAX: int = 2 ** 63 - 1

    def __init__(
        self,
        value: int,
        min_value: int | None = None,
        max_value: int | None = None,
        occurrence_weight: int | None = None,
        deviation_weight: int | None = None,
    ):
        super().__init__(
            occurrence_weight=occurrence_weight,
            deviation_weight=deviation_weight,
        )

        self._value: int = value
        self.min_value: int = max(min_value or self.MIN, self.MIN)
        self.max_value: int = min(max_value or self.MAX, self.MAX)

        assert self.MIN <= self.min_value <= self.value <= self.max_value <= self.MAX

    @property
    def value(self) -> int:
        return self._value

    @value.setter
    def value(self, value: int):
        self._value = max(self.min_value, min(self.max_value, value))

    def mutate(self, num: int):
        sign = randint(0, 1) * 2 - 1
        deviation = randint(1, self.deviation_weight * num)
        self.value += sign * deviation

class UintMutagen(IntMutagen):
    MIN: int = 0
    MAX: int = 2 ** 64 - 1

class Mutator:
    def __init__(self, mutagens: list[Mutagen]):
        self.mutagens: list[Mutagen] = mutagens
        self.mutagen_map = {i: m for i, m in enumerate(self.mutagens)}
        self.mutation_selector = [i for i, m in self.mutagen_map.items() for _ in range(m.occurrence_weight)]

    def mutate(self, num_mutations: int | None = None):
        assert num_mutations > 0

        num_mutations = num_mutations or 1
        shuffle(self.mutation_selector)
        mutations = {}

        for i in range(num_mutations):
            selection = self.mutation_selector[i % len(self.mutation_selector)]

            if selection in mutations:
                mutations[selection] += 1
            else:
                mutations[selection] = 1

        for i, num in mutations.items():
            self.mutagen_map[i].mutate(num)

class Dna:
    def __init__(
        self,
        min_neurons: int = 1,
        n_synapse: int = 1,
        n_dimension: int = 1,
        activation_level: int = 1,
        max_neuron_strength: int = 1,
        refactory_period: int = 0,
        facilitation_granularity: int = 1,
        facilitation_limit: int = 1,
    ):
        # Lower limit on the number of neurons in the hypercube
        self._min_neurons: UintMutagen = UintMutagen(min_neurons, 1)

        # Number of synapses per neuron
        self._n_synapse: UintMutagen = UintMutagen(n_synapse, 1)

        # Number of virtual spacial dimensions in a mind
        self._n_dimension: UintMutagen = UintMutagen(n_dimension, 1, 5)

        # Level at which a neuron activates
        self._activation_level: UintMutagen = UintMutagen(activation_level, 1)

        # Neurons cannot propagate more potential than this when activating
        self._max_neuron_strength: UintMutagen = UintMutagen(max_neuron_strength, 1)

        # Amount of time a neuron stays in the refactory state (min 1)
        self._refactory_period: UintMutagen = UintMutagen(refactory_period)

        self._facilitation_granularity: UintMutagen = UintMutagen(facilitation_granularity, 1)
        self._facilitation_limit: UintMutagen = UintMutagen(facilitation_limit, 1)

        self.mutator: Mutator = Mutator([
            self._min_neurons,
            self._n_synapse,
            self._n_dimension,
            self._activation_level,
            self._max_neuron_strength,
            self._refactory_period,
            self._facilitation_granularity,
            self._facilitation_limit,
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
    def refactory_period(self) -> int:
        return self._refactory_period.value

    @property
    def facilitation_granularity(self) -> int:
        return self._facilitation_granularity.value

    @property
    def facilitation_limit(self) -> int:
        return self._facilitation_limit.value

    def mutate(self, num_mutations: int | None = None):
        self.mutator.mutate(num_mutations)
        self.initialize()
