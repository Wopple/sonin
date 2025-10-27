from pydantic import BaseModel, Field

from sonin.model.hypercube import Vector

# Identifies a signal
type Signal = int

# Number of signals
type SignalCount = int

# Degree of attraction between signals. Negative values are repulsive.
type Affinity = int

# How attracted a top level signal is to the inner signals
type AffinityDict = dict[Signal, dict[Signal, Affinity]]


class SignalProfile(BaseModel):
    affinities: AffinityDict = Field(default_factory=dict)

    def attraction(self, this_signal: Signal, that_signal: Signal) -> Affinity:
        attraction = self.affinities.get(this_signal)

        if attraction:
            return attraction.get(that_signal, 0)
        else:
            return 0

    def attraction_force(
        self,
        this_signal: Signal,
        that_signal: Signal,
        this_position: Vector,
        that_position: Vector,
        factor: int = 1,
    ) -> Vector:
        degree_of_attraction = self.attraction(this_signal, that_signal)
        distance = this_position.city_distance(that_position)

        if distance == 0:
            return Vector(
                value=tuple(0 for _ in range(this_position.n_dimension))
            )

        direction = that_position - this_position

        # dividing by distance to weaken farther signals
        # dividing by the square because the first division merely counteracts the amplifying effect of distance when
        # calculating the direction vector
        # multiplying by a factor to avoid loss of precision due to integer division
        return (direction * (factor * degree_of_attraction)) // (distance * distance)
