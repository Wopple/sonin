from datetime import timedelta

from sonin.model.dna import Dna
from sonin.model.evolution import Activity, PetriDish
from sonin.model.fate import BinaryFate, Fate, FateTree
from sonin.model.hypercube import CubeShape, Vector
from sonin.model.mind import Mind
from sonin.model.mutation import DnaMutagen
from sonin.model.neuron import TetanicPeriod
from sonin.model.signal import SignalProfile
from sonin.model.stimulation import SnapBack, Stimulation
from sonin.model.storage import load_samples_local, save_samples_local
from sonin.sonin_random import seed


# Rules
#   Conceptually, neurons and synapses are agents making decisions on their own based on interactions and environment.
#   The mind is the programmatic engine that facilitates that behavior.
#   Floats are 100% banned for performance critical code, even for intermediate values. This is to avoid floating
#   point imprecision, and so specialized hardware does not require any circuitry for performing floating point math.
#   Everything should use ints. I also do not see a need for strings, but strictly speaking they are not banned.
#   Floats can only and strings should only exist outside the mind's interface.

# Reading
#   https://nba.uth.tmc.edu/neuroscience/m/s1/index.htm
#   https://pmc.ncbi.nlm.nih.gov/articles/PMC8186004/
#   https://pmc.ncbi.nlm.nih.gov/articles/PMC4743082/
#   https://biology.kenyon.edu/courses/biol114/Chap11/Chapter_11.html
#   https://www.science.org/doi/10.1126/science.aax6239

# Features
#   + Neurons can be excitatory or inhibitory.
#   + Neurons can detect levels of stimulation.
#   + Synapses are entities as well.
#   + Synapses are bidirectional.
#   + Synapses can strengthen and weaken.
#   + Neurons can regulate excitation when overstimulated by weakening connections.
#   + Longer duration stimuli can lead to the initiation of multiple action potentials. The frequency is dependent on
#       the intensity of the stimulus.
# 
#   - Simple circuits can create behavior like lateral inhibition creating edge enhancement.
#   + Some neurons periodically activate in bursts even without excitation.
#
#   + Neurons that fire together, wire together.
#   - Connected neurons that fail to activate together weaken their connection over time.
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
#     - Concentration gradients can have thresholds that affect the fate of cells.
#     - Sequential induction can affect the fate of cells through one of the signaling mechanisms to neighbors.
#     - Can develop new signals by cloning successful signals and varying them independently.
#     + Signal exposure is inversely proportional to distance.
#     - Local signals could have a maximum range and be all or nothing.
#
#   - Axon guidance is affected by:
#     + both attractive and repulsive forces
#     - near factors
#     + far factors
#     + guidance cells (implemented as signals from cell division)
#     - following existing axons
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

# Model
#   - Dependency Graph
#     - PetriDish > Mutagen
#     - PetriDish > MindInterface
#     - MindInterface > Mind
#     - Mutagen > Mutable Models
#     - Incubator > Hypercube
#     - Mind > Hypercube
#     - Mind > Neuron
#     - Mind > Vector
#     - Mind > Synapse
#     - Neuron > Synapse
#     - Synapse > Vector
#     - Facilitation > Gear

def run_and_plot(sample: DnaMutagen):
    seed(1)
    dna: Dna = sample.value

    mind: Mind = dna.build_mind()
    mind.print_activations = True
    mind.randomize_potential()

    def plot_synapses():
        import matplotlib.pyplot as plt

        line_segments = [
            (s.pre_neuron.value, s.post_neuron.value)
            for n in mind.neurons.items
            for s in n.post_synapses.values()
        ]

        fig, ax = plt.subplots()

        for segment in line_segments:
            (x1, y1), (x2, y2) = segment
            ax.plot([x1, x2], [y1, y2], marker='o')

        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Synapses')
        plt.show()

    for i in range(100):
        mind.step(i)

    plot_synapses()


def evolve(samples: list[DnaMutagen], name: str):
    petri_dish = PetriDish(
        samples=[(s, 0) for s in samples],
        coach=Activity(),
        sample_retention=4,
        num_descendants=4,
        num_mutations=64,
        input_shape=CubeShape(size=2),
        output_shape=CubeShape(size=2),
        reward_shape=CubeShape(size=2),
        punish_shape=CubeShape(size=2),
    )

    petri_dish.evolve(
        initial_sample=DnaMutagen(),
        min_generations=50,
        min_elapsed_time=timedelta(minutes=15),
    )

    save_samples_local(name, [s for s, _ in petri_dish.samples])


if __name__ == '__main__':
    name = 'progress2'
    samples = load_samples_local(name)
    # samples = [DnaMutagen()]
    # evolve(samples, name)
    run_and_plot(samples[1])
