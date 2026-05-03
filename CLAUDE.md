# Sonin

Experimental neural-network platform by Daniel Tashjian. The thesis: Multi-Layer Perceptron has been the foundation of every modern advance in AI for sixty years; its two structural limits (no cycles, stateless neurons) make some kinds of intelligence either impossible or wildly inefficient to express. Sonin abandons MLP entirely and instead simulates a (heavily simplified) biological brain — directed cyclic graph of stateful neurons, action potentials, Hebbian plasticity, optimized by a genetic algorithm rather than gradient descent.

The author's own status, in `README.md`: **"Status: not actively developed"** and **"So... does it work? Not really."** That framing matters — this codebase is a research sketch. The skeleton works, the optimization loop runs, but it has not been shown to learn any useful task. The README explicitly identifies the bottleneck as **the quality of mutations** to the DNA. Reading the next section is essential before solving any "fundamental problem" — the README itself states the problem.

---

## Layout

```
sonin/
  main.py              Entry point — run_and_plot (visualize one mind) and evolve (genetic loop)
  sonin_math.py        div() (rounds toward zero, like Rust) and most_significant_bit()
  sonin_random.py      XorShift32, Pcg32, and Random/HasRandom helpers — Rust-parity RNG
  model/
    dna.py             The genome — hyperparameters that define a Mind
    fate.py            Cell fate (excitatory? axon offset? activation level? tetanic? …)
    paint.py           Spatial shapes that "paint" Fates onto neurons in the hypercube
    hypercube.py       N-dimensional grid; Vector (value + index) + Position abstractions
    mind_factory.py    Builds a Mind from a DNA via the paint→fate process
    mind.py            The brain: step(), propagate_potential, Hebbian learning, MindInterface
    neuron.py          Neuron + Axon + TetanicPeriod (potential, refractory, recent_activations)
    synapse.py         Synapse (pre/post Vector, strength)
    stimulation.py     SnapBack relaxation math + Stimulation wrapper
    facilitation.py    Modulation gear (built but not wired into Mind/Neuron yet)
    gear.py            Gear(up, down, current) — int ratio with carry to avoid precision loss
    step.py            HasStep mixin (step / cleanup)
    metric.py          Metric, SlidingFrequencyProfile (mean & instability)
    lesson.py          Lesson enum + LessonPlan (coaches → mutator)
    mutation.py        Mutator + 26 Mutagen classes that mutate every field of the DNA
    evolution.py       Sample, Coach, Coaches, Health, Echo, PetriDish (the GA loop)
    storage.py         JSON dump/load for DNA into local/<name>.json
tests/
  sonin/
    test_sonin_math.py, test_sonin_random.py
    model/test_facilitation, _gear, _hypercube, _metric, _mind, _neuron, _paint, _stimulation
pyproject.toml         Poetry; Python ≥3.12; matplotlib + pydantic; pytest
README.md              Conceptual overview, intent, failure mode, "next steps if I were to continue"
local/                 Gitignored — genetic samples are saved/loaded here as JSON
```

---

## Design Philosophy (read carefully)

Several non-obvious commitments shape every file. Violating them would be a regression even if technically correct.

**1. Integers everywhere. No floats.** `sonin_math.div` rounds toward zero (Python's `//` rounds toward `-inf`); `Gear` carries a remainder so an int ratio doesn't lose precision; SnapBack damping is `value * damper // rate`; even fitness is an `(int, int)` lex tuple over int products of measurements. Reasoning: the design targets eventual Rust port (see "Rust parity" below) and integer-only math is deterministic, fast, and reproducible. **Every mutation, every fitness measurement, every neuron state is an int.** When in doubt, do not introduce floats.

**2. Rust parity.** `sonin_random.py` reimplements XorShift32 and PCG32 in Python so a future Rust port produces *bit-identical* RNG sequences. `sonin_math.div` exists solely because Python's `//` differs from Rust's `/` for negatives. The architecture is shaped by: "could a Rust port produce identical output?"

**3. Reproducibility is a hard requirement.** Every `Random` instance is threaded explicitly. `Pcg32().seed(1)` is called inside `PetriDish.evolve` for each mind build (`evolution.py:443`) so the same DNA always builds the same mind. The git log includes `9671ddb fixed reproducibility bugs` and `f8805f2 implemented RNG to control for behavior reproducibility` — this has been hard-won. Do not introduce hidden uses of `default_rng` inside core paths.

**4. Pydantic for serialization.** Every domain object subclasses `BaseModel`. `model_post_init` is used for derived state (e.g. `Vector.index` is computed from `value` + `dimension_size`). DNA is JSON-roundtrippable through `storage.py`. The `Field(exclude=True)` markers (e.g. `TetanicPeriod.dormant`, `TetanicPeriod.n_time`) keep transient state out of the saved DNA.

**5. Two-pass step semantics.** `Mind.step` deliberately separates "advance neurons" from "propagate potential" so reads and writes happen in different phases. `mind.py:135` says: *"This makes the algorithm trivial to parallelize at a later time."* If you rewrite step(), preserve the read/write barrier between phases.

**6. Biology as the spec.** `main.py` lines 12–113 are an unstructured but exhaustive list of biological mechanisms with `+` (implemented) / `-` (not yet) markers. This is the intended feature set — most `-` items are deliberately deferred (electrical synapses, neurotransmitter diffusion, chronic excitatory↔inhibitory flipping, gradient-based positional determination). Don't strip them from `main.py` — they're the long-term roadmap.

**7. Coaches steer mutations via Lesson Plans.** A novel-ish part of the design: a `Coach` doesn't just emit fitness — it emits a `LessonPlan` (`{Lesson.MORE_ACTIVATION: Gear(up=8), …}`) which the `Mutator` consults to weight which mutagens to apply. So a mind that's underactive will preferentially have mutations that increase synapse count, decrease thresholds, etc. This is the project's substitute for "directionality" that gradient descent provides.

---

## Core algorithms

### `Mind.step(c_time)` — `mind.py:130`

Per step, in this order:

1. **Per-neuron pass 1:** call `neuron.step(c_time)` (advances Stimulation snap-back, TetanicPeriod, shifts `recent_activations` left and masks to 64 bits). Activates the neuron if `potential ≥ activation_level` *or* tetanic period says it's time. Mind tracks `num_activations` and a per-64-neuron bitmap `activation_set` for variance metrics.
2. **Overstimulation regulation** (also in pass 1): if a neuron's stimulation exceeds its threshold and has any pre-synapses, find its most-stimulated pre-synaptic input. If that input excites, weaken the connection by `max_synapse_strength/2`; if it inhibits, strengthen it. This is the "homeostatic" rule.
3. **Per-neuron pass 2:**
   - If the neuron is activated, propagate its `synapse.strength` to all post-synaptic neurons (positive if excites, negative if inhibits). Only `ACCEPTING` neurons receive — `REFACTORY` ones are skipped.
   - On every 64th step (using `c_time % 64 == n.position.index % 64` as a hash so different neurons get serviced on different ticks), run the **Hebbian** rule `strengthen_simultaneous_activations`: for every neuron in axon range that this neuron's recent activations *preceded* (correlated by 1 step), either strengthen an existing synapse (+1) or form a new one if under `max_synapses`.
4. **Cleanup** (`Mind.cleanup`) deactivates everything in preparation for the next tick.

The whole "mind" is run in a stream — the `evolve` loop runs ~64 ticks per coaching session (`Health.d_time = 64`, `Echo.d_time = 64`). One mutation generation evaluates one mind for ~64 steps.

### `PetriDish.evolve(...)` — `evolution.py:410`

```
descendants ← initial_samples
while not (generations done and time elapsed):
    if every descendant has measurements (i.e. not the first round):
        descendants ← []
        for each retained sample:
            for _ in range(num_descendants):
                child_dna ← deep copy of sample.dna
                Mutator.mutate(child_dna, num_mutations, sample.lesson_plan)
                descendants.append(sample.build_next(child_dna))
    for each descendant:
        rng ← Pcg32(); rng.seed(1)                      # reset every time, reproducible
        mind ← MindFactory(descendant.dna).build_mind(Random(rng))
        mind.randomize_potential()
        coach.mind = mind; coach.sample = descendant
        coach.reset(); c_time = 0
        while not coach.done:
            coach.step(c_time); coach.cleanup(c_time); c_time += 1
        descendant.measurements, descendant.lesson_plan ← coach.measure()
        new_samples.append((descendant, lesson_plan))
    self.samples ← heapq.nsmallest(retention, new + old, key=fitness)   # keep best N
    print((generation, top samples))
```

Smaller fitness wins. `Sample.total_fitness` returns a lex tuple `(health_product, task_product)` and tuples compare lexicographically. Each `Coach` declares `kind: 'health' | 'task'` and `measurement_arity: int`; `Coaches` aggregates these into a `health_mask` parallel to the measurement tuple, so `Sample` knows which axes are which.

Three layers of "tolerance" stack on top of the raw measurements:

1. **Parent-relative noise snap** (`evolution.py in_tolerance`) — if a child's measurement is within `[baseline, baseline + 1 + msb(baseline)]` of its parent's, the parent's value is substituted before any further math. Absorbs single-step jitter so it doesn't propagate.
2. **Per-axis health bands** (`Sample.tolerances`) — each health axis contributes `1 + max(0, m − tolerance_i)` to the health product, so an axis inside its band contributes the multiplicative identity (factor 1). Task axes are multiplied separately as the secondary key.
3. **Plateau-gated band freeze** (`PetriDish.health_patience`, default 64) — bands are *not* set initially. While health is still improving on any axis, `tolerances=None` and the health product reduces to `prod(1 + m_i)` — pure health-driven selection. After `health_patience` generations with no per-axis improvement, the bands freeze at `best_measurements + 1 + msb(best)`. Only then do healthy axes drop to factor 1, allowing the task product to discriminate. This is a deliberate two-phase regime: optimize health first, then optimize task while staying inside the discovered bands.

`Sample.total_fitness` is a regular `@property` (not `cached_property`) because retained samples are re-ranked under updated tolerances each generation.

### Sample → DNA → Mind pipeline — `mind_factory.py:11`

A `Dna` is `(num_dimensions, dimension_size, max_synapses, max_synapse_strength, max_axon_range, fate_paints)`. `fate_paints` is a list of `(Shape, Fate)` pairs. To build a mind:

1. Start with `fate_positions = [None] * num_neurons`. The first paint **must** be `(FillShape(CompleteFill), default_fate)` (asserted in `Dna.model_post_init`) so every neuron gets *some* fate.
2. **Iterate paints in reverse** — later paints in the list act as overlays that *override* earlier fates. The reverse iteration means the first painter wins for each cell (last-in-list = base layer, first-in-list = top overlay). `mind_factory.py:19-30`
3. Build `Neuron`s from the assigned fates (axon position = clipped (position + fate.axon_offset)).
4. Wrap in a `Hypercube[Neuron]`, then a `Mind`, then a `MindInterface` (which adds input/output/reward/punish neuron lists from input/output Shapes anchored at canonical corners — input at all-zero, output opposite, reward at the half/half corner, punish opposite reward; see `mind_factory.py:67-105`).
5. `mind.randomize_synapses()` — for each neuron, pick `rand_int(0, max_synapses)` random nearby positions in axon range and connect with strength `max_synapse_strength/2`.

---

## The DNA — what's actually being evolved

`Dna` (`dna.py`) holds:
- `num_dimensions`: int (default 2)
- `dimension_size`: int (default 6) → `num_neurons = dimension_size ** num_dimensions` (default 36)
- `max_synapses`: int → cap on each neuron's post-synapse count (default 1, mutated up to `MAX_SYNAPSES = 8` in `mutation.py:16`)
- `max_synapse_strength`: int → upper bound for strength (default 1)
- `max_axon_range`: int → axon's spatial reach (default 1, capped at `dimension_size/3`)
- `fate_paints`: `list[(Shape, Fate)]` → the genetic "code" for cell types
- `input_shape`, `output_shape`, `reward_shape`, `punish_shape`: spatial regions (defaults to 3-wide rectangles at canonical corners)

A `Fate` (`fate.py`) is:
- `excites: bool` (excitatory vs inhibitory)
- `axon_offset: tuple[int, ...]` (where the axon points relative to the neuron's position)
- `activation_level: int ≥ 1` (potential threshold)
- `refactory_period: int ≥ 0` (steps inactive after firing)
- `stimulation: Stimulation` (snap-back tracker)
- `overstimulation_threshold: int ≥ 1`
- `tetanic_period: TetanicPeriod` (enabled? threshold? activations? gap?)

A `Shape` is one of `FillShape | RectangleShape | CityShape`, each combinable with a `Fill` (`CompleteFill | ModuloFill | OffsetFill`). Shapes can be `outline=True` (just the border), `wrap=True` (toroidal). `Position` (where the shape is centered) is `AbsPosition | RelPosition`. RelPositions encode `(numerator, delta)` where the actual coord = `dimension_size * numerator // (numerator + delta)` — this is dimension-size-invariant, which means a DNA evolved at `dimension_size=6` has at least some hope of generalizing to other sizes.

---

## The Lesson Plan system (the key biology→GA bridge)

The `Lesson` enum (`lesson.py:13`) currently has only **four** values:
```
MORE_ACTIVATION  LESS_ACTIVATION  MORE_AXON_MOVEMENT  LESS_AXON_MOVEMENT
```

A `LessonPlan` is `dict[Lesson, Gear]`. Looking up a missing lesson yields `identity = Gear(up=1, down=1)`. Looking up a tuple of lessons multiplies their gears (with GCD reduction). When a mutagen wants to weight a candidate mutation, it does e.g.:

```python
# from MaxSynapsesMutagen
weights = [
    (0, lesson_plan[Lesson.MORE_ACTIVATION](inc)),  # increase weight if mind needs more activation
    (1, lesson_plan[Lesson.LESS_ACTIVATION](dec)),
]
```

So the `Health` coach noting "this mind has too many activations" causes the mutator to preferentially apply mutations that reduce activation. This is the project's substitute for gradient descent's directionality. **The four-Lesson vocabulary is small** — only activation count and axon distance get steering; everything else (synapse strength, paint shape, refractory periods, tetanic config, etc.) is mutated with constant weights regardless of the mind's deficiencies.

---

## The Mutator (mutation.py — 997 lines)

A tree of `Mutagen` classes. Each `mutate()` method:
1. Builds a list of `(action_id, weight)` pairs, where weights are scaled by the lesson plan.
2. `weighted_choice` picks an action.
3. Match-statement dispatches to the chosen `inc_x` / `dec_x` / `flip_x` / `update_x` method.
4. For sub-objects, recurses into a child mutagen.

Top-level shape:
```
Mutator
├── MaxSynapsesMutagen           (steered by MORE/LESS_ACTIVATION)
├── MaxSynapseStrengthMutagen    (steered by MORE/LESS_ACTIVATION)
├── MaxAxonRangeMutagen          (no steering)
└── FatePaintsMutagen            (the bulk of it, weighted 16× the others)
    ├── add / remove / swap / replace_paint
    ├── ShapeMutagen
    │   ├── FillShapeMutagen
    │   │   └── FillMutagen → ModuloFillMutagen / OffsetFillMutagen
    │   ├── RectangleShapeMutagen (center, sizes, outline, wrap, fill)
    │   └── CityShapeMutagen      (center, size, outline, wrap, fill)
    └── FateMutagen
        ├── flip_excites
        ├── update_axon_offset    (steered by MORE/LESS_AXON_MOVEMENT)
        ├── update_activation_level
        ├── update_refactory_period
        ├── StimulationMutagen     (steered by MORE/LESS_ACTIVATION)
        ├── update_overstimulation_threshold
        └── TetanicPeriodMutagen   (steered by MORE/LESS_ACTIVATION)
```

The `IntOps` mixin (`mutation.py:136`) is the workhorse: `add` / `sub` / `update_int(value, min, max)` adds or subtracts `rand_int(1, deviation_weight)` and clips. `TupleIntOps.update_tuple_int` mutates one component of a tuple at random. `BASE_WEIGHT = 16` is the per-mutation base; many mutagens use `base_weight / 4` for boolean flips and `base_weight * 16` for fate_paints to dominate the top-level mix.

---

## Coaches & fitness

The fitness function is the `Coach`. Every concrete `Coach` declares two required class attributes (no defaults on the base — subclasses *must* set them):

- `kind: str` — `'health'` or `'task'`. Health axes go into the primary fitness key (with bands); task axes go into the secondary key.
- `measurement_arity: int` — number of ints returned by `measure()`. Used by `Coaches` to build a parallel `health_mask`.

The codebase ships two:

- **`Health`** — `kind='health'`, `measurement_arity=8`. The always-on baseline. Targets:
  - `target_activations = num_neurons / 4` per step (penalizes over- and under-firing)
  - **Activation variance** — the same activation pattern from one step to the next is penalized (`activation_variance_miss`, `activations_set_counts`). Prevents trivial oscillators winning.
  - **Axon load** — one axon-target per cell of `dimension_size` (penalizes everyone pointing the same way; `axon_load_component`)
  - **Axon variance** — same relative axon vector across many neurons is penalized
  - **Axon distance** — total axon length should be roughly `dimension_size`. Emits MORE/LESS_AXON_MOVEMENT lessons.
  - **Paint mean** — number of cells painted per layer should average ≈ `num_neurons/8` (penalizes paints that cover everything or nothing)
  - **Paint instability** — variance across the `paint_count_metric` (penalizes paints that cover wildly different amounts each layer)

- **`Echo`** — `kind='task'`, `measurement_arity=1`. Toy task: each step, fire random input bits; expect the same bits at the output 2 steps later; count bit mismatches.

`Coaches([Health(), Echo()])` runs them concurrently — measurements are concatenated, lesson plans merged, and `Coaches.health_mask` is the per-child mask concatenated. `Coaches` itself does not declare `kind` (it's not a leaf — `health_mask` is overridden to aggregate from children).

---

## Spatial / hypercube layer

`Vector` (`hypercube.py:10`) is the omnipresent type: a `tuple[int, ...]` with a `dimension_size` and a precomputed flat `index`. It supports `+ - * /` (scalar and componentwise), `clip()` (to dimension bounds), `city_distance` (taxicab), `city_unit` (a sketchy approximation of "the city-block unit vector" — `if abs(c) >= largest/2 then sign(c) else 0`).

`Hypercube[T]` is the storage type — flat list of N^D items, addressed by `Vector.index` or by tuple/int. `center()` gives the center cell(s), special-cased for odd vs even `dimension_size`.

`Position` is `AbsPosition | RelPosition`. `RelPosition.value: list[(numerator, delta)]` encodes percentage along each dimension as `dim_size * numerator // (numerator + delta)`. This is what enables paints to remain meaningful as `dimension_size` mutates.

`Shape` (`paint.py`) emits the set of `Vector`s a paint covers: `FillShape` covers the entire grid (potentially with a `ModuloFill` or `OffsetFill` filter); `RectangleShape` covers a centered axis-aligned box; `CityShape` covers a taxicab disk.

---

## Tests

The test suite (1,099 lines, 6 of 18 source modules with non-trivial coverage) thoroughly validates **mathematical primitives** and **isolated component behaviors**, but **completely skips the cognitive loop and evolutionary dynamics**. From the survey:

| Module | Coverage |
|---|---|
| `sonin_math` | `div()` parametrized over sign combinations |
| `sonin_random` | `rotate_right_32`; `rand_bool` for 6 specific seeds (locked-in determinism) |
| `gear` | `Gear.__call__` parametrized — int ratio with carry |
| `facilitation` | `modulate()` exhaustive — 36 parametrized cases |
| `metric` | `SlidingFrequencyProfile.mean` and `instability` |
| `stimulation` | `SnapBack.step` (one tick of relaxation), no convergence tests |
| `neuron` | `TetanicPeriod` state machine only — no Neuron tests |
| `hypercube` | `Vector` arithmetic, `Hypercube.initialize` & `center`. **Gap:** no `Hypercube.get` tests, no negative-coord centering tests |
| `paint` | Big — 35 parametrized cases for RectangleShape & CityShape, including outline+wrap interactions |
| `mind` | **Only** `strengthen_connection` / `weaken_connection`. No `Mind.step`, `propagate_potential`, `strengthen_simultaneous_activations`, no `MindInterface` |
| `evolution` | **No tests** — `PetriDish.evolve`, `Health`, `Echo`, `Sample.total_fitness` are all unverified |
| `mutation` | **No tests** — none of the 26 Mutagen classes. There used to be a `test_mutation.py`; it was deleted in the Nov 2025 refactor (`4a6f542 mutation refactor`) |
| `mind_factory` | **No tests** — paint→fate iteration, axon clipping, input/output anchoring all unverified |
| `dna`, `fate`, `synapse`, `lesson`, `step`, `storage` | **No tests** |

Run tests: `poetry run pytest` (or `.venv/bin/pytest`). `pythonpath = sonin` is configured in `pyproject.toml`. Tests are deterministic; no fuzz/property-based tests.

---

## Known bugs (verified)

These are real defects, not stylistic gripes. Helpful to know before reasoning about behavior.

(A handful of older bugs from an earlier survey have been fixed: `MindInterface.input_indices` was a no-op `set.union` that's now `|=`; `Echo.post_step` was counting matched bits instead of mismatched and is now `count('1')`; `TetanicPeriodMutagen.dec_activations` was calling `.add` instead of `.sub`. Don't re-flag these.)

**1. `MindInterface.activate_by` over-clips by one bit** — `mind.py`, in `activate_by`.
```python
value = min(value, (2 << len(neurons)) - 1)
```
`2 << N` is `2^(N+1)`; should be `1 << N` for `2^N`. With N input neurons, max representable is `2^N - 1`, but the cap allows `2^(N+1) - 1`, i.e. extra phantom bits that have no neuron to fire. Mostly harmless (the high bits get masked off by the loop bound), but indicates the comment doesn't match the math.

**2. `random_relative_coordinate` is biased to non-zero** — `mutation.py`.
```python
numerator = random.rand_int(0, deviation_weight)
delta = random.rand_int(0, deviation_weight)
if numerator == 0 and delta == 0:
    if rand_bool(): numerator = rand_int(1, deviation_weight)
    else: delta = rand_int(1, deviation_weight)
```
This is intentional (RelPosition with both 0 is invalid since `0/(0+0)` divides by zero), but it skews the distribution — low values of `deviation_weight` will see `(0,0)` resampled often, leading to a bimodal distribution. Not a bug per se but worth knowing if you tune `deviation_weight`.

**3. `Vector.clip()` upper-bound case is dead code** — `hypercube.py`.
```python
elif value >= upper:
    return self.dimension_size - 1   # uses self.dimension_size, ignores `upper`
```
The function takes a `lower, upper` argument but only uses `upper` for the comparison; the return value bypasses it. Currently every caller passes `(0, dimension_size)`, so this is harmless but surprising.

**4. `Mind.randomize_synapses` may make zero synapses** — `mind.py`, in `randomize_synapses`.
```python
for i in range(self.rand_int(self.max_synapses)):
```
`rand_int(N)` returns `[0, N]` inclusive, so when `max_synapses=1` the inner loop runs 0 or 1 times. With the default DNA (`max_synapses=1`), about half of all initial neurons have *no outgoing synapses at all*, and the fate-painting first generation often produces minds that can do nothing. This matches what one would observe — the first generation has very low activation, prompting `MORE_ACTIVATION` mutations. Whether that's desired or just luck is unclear. (Separately, `random_position` can legitimately return `None` when an axon's neighborhood is fully covered by input/reward/punish neurons; `randomize_synapses` now early-`continue`s in that case.)

**5. `FatePaintsMutagen.update_paint` shadows variable but never uses `fate`** — `mutation.py`.
```python
paint, fate = dna.fate_paints[update_idx]
self.paint.mutate(dna, lesson_plan=lesson_plan, subject=paint)
```
Cosmetic — `fate` unpacked but unused.

---

## The fundamental problems (per the README, validated by code reading)

The README is upfront about what's broken:

> The DNA needs a mechanism that balances these properties:
> - expressiveness of the primitives (the full set of possible intelligences should be supported)
> - impactful changes when mutated (each mutation should make a noticeable difference)
> - controlled changes when mutated (each mutation should remain "near" the current state)

The Echo / input-isolation / `dec_activations` bug fixes plus the rewrite of fitness as a lex `(health, task)` tuple with adaptive bands have all landed, and the README's "does it work? Not really" verdict still stands. The remaining concrete weaknesses, after a code reading:

**A. The Lesson vocabulary is too narrow.** Only `MORE/LESS_ACTIVATION` and `MORE/LESS_AXON_MOVEMENT` exist. The Health coach has 8 measurement components but emits guidance for only 2 of them (axon distance, indirectly activation). Things like "the mind has insufficient activation variance," "paints are too uniform," "synapses cluster too tightly" — none of these become lessons. So the mutator wanders blindly with respect to most defects.

**B. Mutations don't compose to interesting structures.** Mutating one tuple component at a time (`TupleIntOps.update_tuple_int`) gives smooth single-axis movement but can't easily produce, say, a mirrored/symmetric paint or a coordinated axon-array. Without crossover or higher-order operators the search explores axis-aligned neighbors only.

**C. Fitness is multi-key now but still not multi-objective.** `total_fitness` is a `(health, task)` lex tuple with a multiplicative product *within* each key. A mind great at 7 of 8 health components and bad at 1 still loses to a mind mediocre everywhere whose product happens to be smaller. Pareto-style selection over the measurement tuple — or at least Pareto within the health key — would preserve more diversity than the current within-key product. The masked product (`1 + excess` per axis after band freeze) addresses the *band* part of the original "single-objective" complaint but not the *Pareto* part.

**D. The `Health` coach optimizes for "looking healthy" not for "doing anything."** `target_activations = num_neurons/4` and "no two consecutive activation patterns may match" is a fine prior, but it has no relationship to performing a task. The two-phase fitness regime (health then task) gives task a real role only after `health_patience` generations of plateau, but Echo is the only available task and it's a toy. Real progress would require a coach with a non-trivial I/O task.

**E. The mind is single-threaded Python over a Python list of Pydantic models.** A 6×6 = 36-neuron grid stepping 64 times runs in seconds, but the README's ambitions (richer minds, longer evaluations) hit a hard wall fast. Pydantic validation on every model construction is non-trivial overhead. The two-pass step structure was *designed* to be parallelizable but isn't.

**F. There is no parent-offspring crossover.** Generations are pure mutation, no recombination. Neuroevolution research generally finds crossover useful for combining good substructures.

**G. No "ageing" / pruning lifecycle.** README mentions "young: enthusiastically form connections / adult: optimize through pruning / elderly: degradation" as planned. Currently every mind is evaluated for the same fixed 64 ticks with no developmental phase.

**H. Empirically, even with the recent fixes the system still does not learn the Echo task.** This rules out "Echo was just inverted" and "input neurons were polluting the network" as the gating issues. The bottleneck is now upstream of those — most likely some combination of A, B, and the underlying mutation-quality concern the README originally flagged.

---

## Strengths (do not regress these)

- **Clean separation** between DNA (genotype), Fate/Paint (developmental rules), Mind (phenotype), Mutator (genetic operators), Coach (fitness). Adding a new biological mechanism almost always touches just two or three of these layers.
- **Pydantic+JSON DNA** — minds can be saved/loaded mid-experiment.
- **Reproducibility discipline** — every run with the same seed and same DNA produces the same Mind, byte-for-byte. The custom RNG and integer math make this work.
- **The lesson-plan idea** — even though the vocabulary is too small, the mechanism for coaches to inform mutators is novel and worth building on.
- **Two-phase `Mind.step` is parallelizable in principle** — read/write are separated; only the overstimulation rule (which mutates pre-synaptic state during pass 1) breaks pure determinism.
- **Mutation tree mirrors data tree** — every Pydantic model has a corresponding mutagen, structured the same way. This makes adding new fields mechanical.
- **Heavy parametrized test coverage on the math primitives** — vector arithmetic, gear ratios, snap-back relaxation, paint geometry are well-pinned. Refactoring the math layers is safe.

---

## Where to look first if asked to "fix the fundamental problems"

In rough priority order (most leveraged first):

1. **Expand the Lesson vocabulary.** Add lessons for paint coverage, paint instability, activation variance, synapse density. Wire them through `Health.measure()` and the existing mutagens. This is mostly mechanical but high-leverage — right now the mutator gets steering on only 2 of the 8 health axes. With the new `(health, task)` fitness in place and Echo no longer inverted, untargeted mutation pressure on most axes is the most likely reason the system still doesn't learn.
2. **Add Pareto (or at least anti-collapsing) selection within the health key.** The current within-key fitness is `prod(1 + excess_i)`, which still lets one bad axis dominate the product. Either Pareto-front retention within the health key or a per-axis rank-sum would diversify the surviving population.
3. **Add a real I/O task coach** that's not Echo (memory? edge detection? simple binary classification?). With the two-phase fitness regime, a real task gets to drive selection only after health plateaus — and Echo is too trivial to demonstrate that the regime works.
4. **Add tests for `Mind.step`.** A handful of small fixed-DNA scenarios with known expected activations would unlock safe refactoring of the cognitive loop. The two-pass / parallelizable structure makes scenario tests cheap to write.
5. **Consider crossover.** Even simple 1-point crossover on `fate_paints` lists could combine useful substructures from two parents — currently generations are pure mutation, no recombination.
6. **Tune `health_patience`.** Default 64 may be too short or too long for the actual plateau dynamics — visualize the per-axis best-over-time and pick a value that reliably catches the elbow.

For deeper changes the README's own self-critique is the best guide: read README.md sections "So… does it work?" and "If I were to continue this" — the author's "search for prior art in the domain of emergent algorithms" line is a hint at where they thought the bigger payoff was.

---

## Glossary

- **Action potential** — a neuron firing; here: `potential ≥ activation_level → activate()`.
- **Excitatory / inhibitory** — sends positive vs negative potential to post-synaptic neurons.
- **Refractory period** — steps after firing during which a neuron cannot fire again. Implemented as `state == REFACTORY` until `c_time >= t_refactory_end`.
- **Tetanic period** — periodic spontaneous bursts of activation independent of input. Encoded by `enabled, threshold, activations, gap`.
- **Synapse** — connection between two neurons with a strength (1 to `max_synapse_strength`).
- **Hebbian plasticity** — "neurons that fire together, wire together." Implemented in `Mind.strengthen_simultaneous_activations` using `(pre.recent_activations >> 1) & post.recent_activations` to detect that pre preceded post by one step.
- **Overstimulation / homeostasis** — neurons that fire too much weaken their excitatory inputs to maintain a stable activation rate. In `Mind.step` after `n.step()`.
- **Paint / Fate** — paint = spatial shape; fate = neuron type. A DNA's `fate_paints` is a list of (paint, fate) pairs that gets resolved in `MindFactory` to produce the per-neuron Fate map.
- **Coach** — a fitness function. Drives the mind through `pre_step`/`post_step` hooks, accumulates measurements, emits a `LessonPlan`.
- **Lesson** — a directional hint from coach to mutator (`MORE_ACTIVATION`, etc.). `LessonPlan` weights mutation choices.
- **Hypercube** — N-D grid of neurons. "Vector" means a position in the grid (not a math vector in the FP sense).
- **Snap-back** — integer relaxation toward a baseline. Used by `Stimulation` to model "recently active" with decay.
- **Gear** — `(up, down, current)`. Multiplies an int by `up/down` while carrying the remainder so a chain of gear applications doesn't lose precision.
- **PCG / XorShift** — RNG algorithms. `Pcg32` is the default; reimplemented to match Rust output.
