from pydantic import BaseModel

from sonin.model.fate import FateTree
from sonin.model.hypercube import Vector
from sonin.model.signal import Signal, SignalCount, SignalProfile


class Dna(BaseModel):
    n_dimension: int
    dimension_size: int
    n_synapse: int
    activation_level: int
    max_neuron_strength: int
    axon_range: int
    refactory_period: int
    environment: list[tuple[Signal, SignalCount, Vector]]
    incubation_signals: dict[Signal, SignalCount]
    signal_profile: SignalProfile
    fate_tree: FateTree
