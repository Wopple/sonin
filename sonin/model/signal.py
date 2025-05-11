from dataclasses import dataclass, field

# Identifies a signal
type Signal = int

# How much of a signal a neuron is exposed to. Affects fate determination.
type Level = int

# Degree of attraction between signals. Negative values are repulsive.
type Affinity = int

# The affinities of the top level signals to the inner signals
type AffinityDict = dict[Signal, dict[Signal, Affinity]]


@dataclass
class SignalProfile:
    affinities: AffinityDict = field(default_factory=dict)

    def attraction(self, this_signal: Signal, that_signal: Signal) -> Affinity:
        attraction = self.affinities.get(this_signal)

        if attraction:
            return attraction.get(that_signal, 0)
        else:
            return 0
