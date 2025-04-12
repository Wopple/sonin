from dataclasses import dataclass

from sonin.model.hypercube import Vector


@dataclass
class Synapse:
    pre_neuron: Vector
    post_neuron: Vector
    strength: int
    max_strength: int

    def strengthen(self, strength: int):
        self.strength = min(self.strength + strength, self.max_strength)
