from sonin.model.hypercube import Position


class Synapse:
    def __init__(self, pre_neuron: Position, post_neuron: Position, strength: int, max_strength: int):
        self.pre_neuron: Position = pre_neuron
        self.post_neuron: Position = post_neuron
        self.strength: int = strength
        self.max_strength: int = max_strength

    def strengthen(self, strength: int):
        self.strength = min(self.strength + strength, self.max_strength)
