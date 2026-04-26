# Sonin

A "failed" attempt at advancing past Multi-layer Perceptron. Sonin is an experimental new neural network platform to
introduce properties of biological minds that Multi-layer Perceptron lacks.

Status: not actively developed

## Context

State-of-the-art AI has become very sophisticated. We have found very interesting ways to leverage neural nets to
exploit their potential. The notable advances have come from building on top of a powerful platform: Multi-layer
Perceptron. MLP isn't new, it dates back at least to the 1960s. What's new is how we use them and the dramatic advances
in hardware.

This leads me to question what comes next? If we build on top of a superior foundation, how much further can AI go?

## Next

It is difficult to come up with something that can compete with MLP:

- MLP has been optimized through decades of research
- MLP is a natural fit for matrix multiplications which fits perfectly with accelerated hardware
- MLP has a powerful optimization semantic with gradient descent

So to compete, we have to look at MLP's weaknesses. I can identify 2 major weaknesses:

1. No cycles in circuitry
2. Stateless neurons

Biological intelligence leverages cycles in circuitry all over the place. Even the simple cycles which are easier to
study can yield some interesting behavior. The more complex circuitry that relies on cycles is likely far more potent.
Cycles likely do not unlock intelligence that cannot be computed otherwise, instead they will unlock far more efficient
ways to perform the computation.

Biological intelligence is also stateful. A neuron receives a signal, that changes its state in both short term and long
term ways. This change in state allows for recognizing object permanence, that car that is hidden behind bushes still
exists in your mind even if there no longer is a visual signal.

Sonin does not try to build upon MLP. The lack of cycles is a hard constraint with how MLP is designed. If you create
just one cycle, back propagation and even forward passes cannot terminate. The statelessness is not quite as hard a
constraint, but it's difficult to imagine how to change the weights during operation without breaking the intelligence
in uncontrollable ways. For these reasons, Sonin starts from scratch.

## Idea

MLP arose by taking cues from biology. Sonin does the same, but it attempts to hew closer to the biology than MLP. Any
attempt at simulating the complete biological process is likely doomed. Biology runs "on the metal" of the universe. It
uses fundamental forces like electro-magnetism at the cellular and even atomic scale to function. Simulating that is
like using a virtual machine in a computer. In theory it can be done, but it will be much slower. If you tried to
capture every force of the universe in a simulation, it would take a preposterous amount of computation to simulate even
the tiniest of minds. So while Sonin does take more cues, it still necessarily takes liberties to simply.

### Directed Cyclic Graph

Sonin uses a DCG. It is not organized into layers. It is organized as a hypercube of neurons where theoretically any
neuron can connect to any other neuron. This is taken directly from biology. Some neurons are designated as input
neurons that are fired by external process, and some neurons are designated output neurons which are read by external
processes. Neurons only connect to a finite number of other neurons.

This is taken directly from biology. Human brains are 3D hypercubes. There are sensory neurons which read information
from the environment, and there are motor neurons which control interaction with the environment through movement.

### Action Potential

Sonin simulates a simplified action potential. When the input potential reaches a threshold, a neuron fires and sends
signal down the post synapses. Neurons can be either excitatory or inhibitory. This is all informed by biological
processes.

### Propagation

MLP inference starts at the input and traverses the DAG until completion. This is possible because there are no cycles.
Sonin works by using input to activate the respective input neurons, then it processes all neurons in parallel. It will
likely have to process the mind multiple times before the input can yield a useful output. The mind can be seen as a
"stream" rather than a "batch."

### Neuroplasticity

There are several kinds of neuroplasticity. Sonin attempts to simulate them. The simplest is Hebbian neuroplasticity. It
is succinctly described as, "neurons that fire together, wire together." It's easy to see the "logic" behind this. If
two neurons tend to fire around the same time, there may be an important dependency there, so establish one by
strengthening connection.

### Optimization

Since backpropagation based gradient descent is out owing to the presence of cycles, there needs to be a different
mechanism for optimization. Once again, biology is the inspiration. Sonin uses a genetic algorithm to optimize the
minds. The mind is defined by something like a DNA (hyperparameters). That definition is mutated randomly in each
iteration, and then the best performing mutations survive to the next stage. There is an attempt to guide mutation by
looking carefully at where performance is lacking and prioritizing mutations that are likely to impact that area of
behavior.

### Other Biological Inspirations

Tetanic Periods: neurons sometimes fire in bursts periodically without any input, useful for a sense of time
Overstimulation: neurons that are overstimulated will weaken input connections to maintain homeostasis
Synaptic Facilitation / Depression: patterns of stimulation can make neurons more or less prone to future stimulation

### Comparison to MLP

**Advantages**
  - Cycles: logical circuits can form cyclic graphs
  - Memory: each neuron is stateful to support memory
  - Neuroplasticity: the connections of neurons can update through activity to support learning from experience
  - Incrementality: can build upon existing models instead of training from scratch
  - Coaching: can perform optimization with little to no data through coaches
  - Customization: there are endless ways to customize the mechanics

**Disadvantages**
  - Slope-less: there is no mathematical slope for gradient descent
  - Slow: each neuron is slower to process so it supports fewer neurons
  - Nascent: does not have 60+ years of research behind it

The main disadvantage is the speed of optimization which is heavily impacted by both the lack of direction from no
backpropagation and the complexity of the neuron. In order to outperform MLP, major optimizations will be necessary, and
each neuron needs to provide much more meaningful intelligence than each neuron in MLP.

## So... does it work?

Not really. I have gotten it to the point where I can get it to optimize for what I consider to be a healthy mind (e.g.
not brain dead, little repetitive behavior, etc.), but I have not gotten it to the point where it can optimize for
useful tasks. To get there will require significantly more research.
