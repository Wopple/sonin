from pydantic import BaseModel, Field

from sonin.model.hypercube import Vector

# Identifies a signal
type Signal = int

# Number of signals
type SignalCount = int

# Degree of attraction between signals. Negative values are repulsive.
type Affinity = int

# How attracted a target signal is to the source signals
# dict[Target, dict[Source, Affinity]]
type AffinityDict = dict[Signal, dict[Signal, Affinity]]


class SignalProfile(BaseModel):
    affinities: AffinityDict = Field(default_factory=dict)

    def attraction(self, target_signal: Signal, source_signal: Signal) -> Affinity:
        attraction = self.affinities.get(target_signal)

        if attraction:
            return attraction.get(source_signal, 0)
        else:
            return 0

    def attraction_force(
        self,
        target_signal: Signal,
        source_signal: Signal,
        target_position: Vector,
        source_position: Vector,
        factor: int = 1,
    ) -> Vector:
        degree_of_attraction = self.attraction(target_signal, source_signal)
        distance = target_position.city_distance(source_position)

        if distance == 0:
            return Vector.of((0,) * target_position.num_dimensions)

        direction = source_position - target_position

        # Dividing by distance to weaken farther signals.
        # Dividing by the square because the first division merely counteracts the amplifying effect of distance when
        # calculating the direction vector.
        # Multiplying by a factor to avoid loss of precision due to integer division and to support multiple signals.
        return (direction * (factor * degree_of_attraction)) // (distance * distance)
