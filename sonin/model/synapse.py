from sonin.model.hypercube import Position


class Synapse:
    def __init__(self, strength: int, pre_neuron: Position, post_neuron: Position | None = None):
        self.strength: int = strength
        self.pre_neuron: Position = pre_neuron
        self.post_neuron: Position | None = post_neuron
