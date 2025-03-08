from random import seed, randint

from model.neuron import ACCEPTING, ACTIVATED
from sonin.model.dna import Dna
from sonin.model.hypercube import Hypercube, Position
from sonin.model.neuron import Neuron

# Rules
#   Conceptually, neurons and synapses are agents making decisions on their own based on interactions and environment.
#   The mind is the programmatic engine that facilitates behavior.

# Reading
#   https://nba.uth.tmc.edu/neuroscience/m/s1/index.htm
#   https://pmc.ncbi.nlm.nih.gov/articles/PMC8186004/
#   https://pmc.ncbi.nlm.nih.gov/articles/PMC4743082/
#   https://biology.kenyon.edu/courses/biol114/Chap11/Chapter_11.html

# Features
#   + Neurons can be excitatory or inhibitory.
#   + Neurons can detect levels of stimulation.
#   - Neurons can regulate excitation when overstimulated by weakening connections.
#   - Longer duration stimuli can lead to the initiation of multiple action potentials. The frequency is dependent on the
#       intensity of the stimulus.
# 
#   - Simple circuits can create behavior like lateral inhibition creating edge enhancement.
#   - Some neurons periodically activate in bursts even without excitation.
#
#   - Neurons that fire together, wire together.
#   - Connected neurons that fail to activate together weaken their connection over time.
#   - Not all synapses exhibit this behavior.
#
#   - There are absolute and partial refactory periods. We may or may not need them though because one if their purposes
#       is to ensure unidirectional activation along an axon which the code already enforces. It's still possible it has
#       important behavioral implications.
#
#   - Neurons gradually return to their resting potential, both temporal and spatial summation can cause a neuron to fire.
#
#   - Some neurons activate for a long period of time based on changes in metabolism.
#
#   - There is a short delay between action potential and the communication to connected neurons. There are gap junction
#       synapses with minimal delay. We may not need to simulate this.
#
#   - Intrinsic synaptic plasticities include:
#     - synaptic depression
#     - synaptic facilitation
#     - post-tetanic potentiation
#   - Synaptic depression reduces the potential passed along when repeats happen with some distance apart.
#   - Synaptic facilitation increases the potential passed when repeats happen a very short distance apart.
#       Continued repeats will have a reduced increasing effect.
#   - Post-tetanic potentiation is a lasting increase in synaptic susceptibility after a high frequency series of action
#       potentials. This is likely related to memory.
#
#   - Synapses can be connected to other synapses to regulate their activity. They can be excitatory or inhibitory
#       creating lasting changes.
#   - Synaptic plasticity depends on the post-synaptic neuron.
#
#   - Neurons also communicate environmentally by diffusing neurotransmitters.
#
#   - Synapses are mostly formed over a short period of time and are mostly pruned slowly over a longer period of time.
#       Interaction with the world happens during both periods.
#
#   - Even in mature minds, synaptic turnover is high estimated at 40% per week but varies a lot depending on the region.
#   - The axon and dendrites tend to remain stable.
#
#   - Neurons mostly use the same neurotransmitter, but some can use multiple or switch. Synapses still only get one.
#   - Synapses will only form such that the neurotransmitter will be received. This could simply be implemented by enum. 
#
#   - Synaptic Adhesion Molecules (SAMs) promote or inhibit both the formation of synaptic connections and their strength.
#   - Pre-synaptic SAMs are more general and post-synaptic SAMs are more specific.
#   - SAMs probably affect the synaptic properties.
#
#   - It is not known if synaptic formation is based on priority or shaped by making eager connections and dropping the
#       ones that are performing poorly.
#
#   - Neurons ought to be able to have hurt-box / hit-box regions to constrain synaptic formation.
#
#   - Positional determination:
#       Concentration gradients can have thresholds that affect the fate of cells.
#       Sequential induction can affect the fate of cells through one of the signaling mechanisms to neighbors.
#
#   - Axon guidance is affected by:
#     - both attractive and repulsive forces
#     - both near and far factors
#     - guidance cells
#     - following existing axons
#
#   - Synapses will compete and favor the more active one.
#   - Activity dependent signals will prune synapses.
#   - Elimination is the default process. Continuance occurs when a cell is provided nutrition.
#   - Synapses retract from undernourished neurons.
#
#   - There needs to be a way to make it unlikely to reform a disconnected synapse.


def random_position(dna: Dna) -> Position:
    return Position(dna.dimension_size, tuple(randint(0, dna.dimension_size - 1) for _ in range(dna.n_dimension)))


class Mind:
    def __init__(self, dna: Dna):
        self.dna: Dna = dna
        self.neurons: Hypercube[Neuron] = Hypercube(dna)
        self.neurons.initialize(lambda position: Neuron(dna, position))

    def randomize_synapses(self):
        for n in self.neurons:
            for i in range(self.dna.n_synapse):
                n.synapses[i] = random_position(self.dna)

    def randomize_potential(self):
        for n in self.neurons:
            if randint(0, 1):
                n.potential = self.dna.activation_level
            else:
                n.potential = 0

    def step(self, c_time: int):
        for n in self.neurons:
            n.step(c_time)

        for pre_n in self.neurons:
            if pre_n.state == ACTIVATED:
                pre_n.refactor(c_time)

                for position in pre_n.synapses:
                    post_n = self.neurons.get(position)

                    if post_n.state == ACCEPTING:
                        if pre_n.excites:
                            post_n.potential += 1
                        else:
                            post_n.potential -= 1


seed(0)

dna = Dna(
    min_neurons=100,
    n_synapse=4,
    n_dimension=2,
    activation_level=3,
)

mind = Mind(dna)
mind.randomize_synapses()
mind.randomize_potential()

input_neurons = mind.neurons.items[:6]
output_neurons = mind.neurons.items[-6:]

def print_neurons(msg: str, neurons: list[Neuron]):
    print(f"{msg}: {[(n.potential, n.stimulation.value) for n in neurons]}")

def print_mind():
    ns = []
    for idx, n in enumerate(mind.neurons.items):
        ns.append(n.potential)

        if len(ns) == 7:
            print(ns)
            ns = []

# print_mind()
print_neurons("input", input_neurons)
print_neurons("output", output_neurons)

for i in range(10):
    mind.step(i)
    print()
    # print_mind()
    print_neurons("input", input_neurons)
    print_neurons("output", output_neurons)

