from datetime import timedelta

import matplotlib.pyplot as plot

from sonin.model.dna import Dna
from sonin.model.evolution import Coaches, Echo, Health, PetriDish, Sample
from sonin.model.mind import Mind, MindInterface
from sonin.model.mind_factory import MindFactory
from sonin.model.storage import load_samples_local, save_samples_local
from sonin.sonin_random import Pcg32, Random, seed

# Reading
#   https://nba.uth.tmc.edu/neuroscience/m/s1/index.htm
#   https://pmc.ncbi.nlm.nih.gov/articles/PMC8186004/
#   https://pmc.ncbi.nlm.nih.gov/articles/PMC4743082/
#   https://pmc.ncbi.nlm.nih.gov/articles/PMC9513053/
#   https://biology.kenyon.edu/courses/biol114/Chap11/Chapter_11.html
#   https://www.science.org/doi/10.1126/science.aax6239

# Features
#   + Neurons can be excitatory or inhibitory.
#   + Neurons can detect levels of stimulation.
#   + Synapses are entities as well.
#   + Synapses are bidirectional.
#   + Synapses can strengthen and weaken.
#   - There are electrical synapses that allow a group of neurons to be activated simultaneously.
#   + There are chemical synapses that have a delay in the transmission of signals.
#   + Longer duration stimuli can lead to the initiation of multiple action potentials. The frequency is dependent on
#       the intensity of the stimulus.
# 
#   - Simple circuits can create behavior like lateral inhibition creating edge enhancement.
#   + Some neurons periodically activate in bursts even without excitation.
#
#   - Not all synapses exhibit this behavior.
#
#   + There are absolute and partial refactory periods. We may or may not need them though because one if their purposes
#       is to ensure unidirectional activation along an axon which the code already enforces. It's still possible it has
#       important behavioral implications.
#
#   + Neurons gradually return to their resting potential, both temporal and spatial summation can cause a neuron to
#       fire.
#
#   - Some neurons activate for a long period of time based on changes in metabolism.
#
#   - There is a short delay between action potential and the communication to connected neurons. There are gap junction
#       synapses with minimal delay. We may not need to simulate this.
#
#   - Neuroplasticity
#     - Synaptic plasticity depends on the post-synaptic neuron.
#       - Connected neurons that fail to activate together weaken their connection over time.
#     + Neurons that fire together, wire together (short timescale).
#     - Neurons self regulate to maintain homeostasis (medium timescale).
#       + Neurons can regulate excitation when overstimulated by weakening connections.
#       - Neurons can regulate excitation when understimulated by strengthening connections.
#     - Some neurons can flip between excitatory and inhibitory due to chronic activation (long timescale). 
#     - Synaptogenesis
#       - Depends upon local state.
#       - Synapses are influenced by neural activities.
#       - Activation (e.g. tetanic) increases the likelihood of forming input synapses.
#       - Synapses are formed enthusiastically at the beginning.
#         - Learning involves synaptic pruning to operate efficiently.
#       - Synaptic Adhesion Molecules (SAMs) promote or inhibit both the formation of synaptic connections and their
#         strength.
#       - Pre-synaptic SAMs are more general and post-synaptic SAMs are more specific.
#       - SAMs probably affect the synaptic properties.
#
#   - Intrinsic synaptic plasticities include:
#     - synaptic depression
#     - synaptic facilitation
#     - post-tetanic potentiation
#   - Synaptic depression reduces the potential passed along when repeats happen with some distance apart.
#   - Synaptic facilitation increases the potential passed along  when repeats happen a very short distance apart.
#     - Continued repeats will have a reduced increasing effect.
#   - Post-tetanic potentiation is a lasting increase in synaptic susceptibility after a high frequency series of action
#       potentials. This is likely related to memory.
#
#   - Synapses can be connected to other synapses to regulate their activity. They can be excitatory or inhibitory
#       creating lasting changes.
#
#   - Neurons also communicate environmentally by diffusing neurotransmitters.
#   - A small percentage of neurons fire with very high variation
#
#   - Synapses are mostly formed over a short period of time and are mostly pruned slowly over a longer period of time.
#       Interaction with the world happens during both periods.
#
#   - Even in mature minds, synaptic turnover is high estimated at 40% per week but varies a lot depending on the
#     region.
#   - The axon and dendrites tend to remain stable.
#
#   - Neurons mostly use the same neurotransmitter, but some can use multiple or switch. Synapses still only get one.
#   - Synapses will only form such that the neurotransmitter will be received. This could simply be implemented by enum. 
#
#   - It is not known if synaptic formation is based on priority or shaped by making eager connections and dropping the
#       ones that are performing poorly.
#
#   - Neurons ought to be able to have hurt-box / hit-box regions to constrain synaptic formation.
#
#   - Positional determination:
#     - Concentration gradients can have thresholds that affect the fate of cells.
#     - Sequential induction can affect the fate of cells through one of the signaling mechanisms to neighbors.
#     - Can develop new signals by cloning successful signals and varying them independently.
#     - Local signals could have a maximum range and be all or nothing.
#
#   - Synapses will compete and favor the more active one.
#   - Activity dependent signals will prune synapses.
#   - Elimination is the default process. Continuance occurs when a cell is provided nutrition.
#   - Synapses retract from undernourished neurons.
#
#   - There needs to be a way to make it unlikely to reform a disconnected synapse so it can try new paths.
#
#   - Most neurons rarely fire, many stay mostly dormant and then see rapid use when needed for a task.
#
#   - Axons can be any length.

# Tests
#   - Weighted suite of tests
#   - Monkey test: resilient in the presence of change
#   - Priority:
#     1. Health
#       - activity within an appropriate range
#       - neuroplastic properties
#       - comprehensive connectivity
#       - differentiation
#       - age appropriate behavior
#         - young: enthusiastically form connections
#         - adult: optimize through pruning
#         - elderly: degradation to prevent runaway
#     2. Optimization
#       - task specialization
#     3. Globalization
#       - intermodule connections
#     4. Competition
#       - minds compete with each other

# Advantages
#   - Cycles: logical circuits can form cyclic graphs
#   - Memory: each neuron is stateful to support memory
#   - Neuroplasticity: the connections of neurons can update through activity to support learning from experience
#   - Incrementality: can build upon existing models instead of training from scratch
#   - Coaching: can perform optimization with little to no data through coaches
#   - Customization: there are endless ways to customize the mechanics

# Disadvantages
#   - Slope-less: there is no mathematical slope for gradient descent
#   - Slow: each neuron is slower to process so it supports fewer neurons
#   - Nascent: does not have 60+ years of research behind it


def run_and_plot(dna: Dna):
    rng = Pcg32()
    rng.seed(1)
    factory = MindFactory(dna)
    mind_interface: MindInterface = factory.build_mind(Random(rng))
    mind: Mind = mind_interface.mind
    mind.print_activations = True
    mind.randomize_potential()

    def plot_synapses():
        line_segments = [
            (s.pre_neuron.value, s.post_neuron.value)
            for n in mind.neurons.items
            for s in n.post_synapses.values()
        ]

        fig, ax = plot.subplots()

        for segment in line_segments:
            (x1, y1), (x2, y2) = segment
            ax.plot([x1, x2], [y1, y2], marker='o')

        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Synapses')
        plot.show()

    def plot_axons():
        line_segments = [
            (n.position.value, n.axon.position.value)
            for n in mind.neurons.items
        ]

        fig, ax = plot.subplots()

        for segment in line_segments:
            (x1, y1), (x2, y2) = segment
            ax.plot([x1, x2], [y1, y2], marker='o')

        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Axons')
        plot.show()

    plot_axons()
    plot_synapses()

    for i in range(32):
        mind.step(i)
        mind.cleanup(i)

    plot_synapses()


def evolve(
    initial_samples: list[Sample],
    name: str,
    min_generations: int,
    min_elapsed_time: timedelta,
):
    petri_dish = PetriDish(
        coach=Coaches([Health(), Echo()]),
        sample_retention=16,
        num_descendants=1,
        num_mutations=1,
    )

    petri_dish.evolve(
        initial_samples=initial_samples,
        min_generations=min_generations,
        min_elapsed_time=min_elapsed_time,
    )

    save_samples_local(name, [s.dna for s, _ in petri_dish.samples])


def main():
    seed(1)
    name = '1'

    samples = [Sample(dna=Dna.from_defaults())]
    # samples = [Sample(dna=dna) for dna in load_samples_local(name)]

    evolve(samples, name, 32, timedelta(minutes=2))
    # run_and_plot(samples[0].dna)


if __name__ == '__main__':
    main()
