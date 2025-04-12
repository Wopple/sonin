type Signal = int
type Level = int
type Affinity = int
type AffinityDict = dict[Signal, dict[Signal, Affinity]]


class SignalProfile:
    def __init__(self, affinities: AffinityDict):
        # A signal is identified by an integer. The affinity maps a signal to how attracted it is to
        # other signals. A negative attractive effect is a repulsive effect.
        self.affinities: AffinityDict = affinities

    def attraction(self, this_signal: Signal, that_signal: Signal) -> Affinity:
        attraction = self.affinities.get(this_signal)

        if attraction:
            return attraction.get(that_signal, 0)
        else:
            return 0
