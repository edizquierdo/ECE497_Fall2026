# ECE497 — Evolutionary Robotics: Project Collection

## Why this sequence exists

Every project in this collection asks a version of the same question: **where does intelligent behavior actually come from?**

You'll start with a robot that has no brain at all — just two wires connecting sensors to motors — and watch it do something that looks purposeful. By the end, you'll have evolved a recurrent neural circuit that discovers, entirely on its own, how to coordinate six legs into a walking gait nobody hand-designed. Nothing in between is a detour. Each project hands you a working piece of machinery and a real, open question about it, and each one builds directly on the last: the fitness functions you write in Project 1 come back in Project 4; the neuroevolution pipeline you build in Project 3 gets reused, largely unchanged, all the way through Project 6; and the genetic algorithm from Project 2 shows up one more time in Project 6 standing next to a second algorithm entirely (CMA-ES), so you can compare them directly instead of taking Project 2's conclusions on faith.

This is also, honestly, a good way to spend a term: you get to watch things evolve. A population of random, flailing controllers turning into something that reliably seeks light, balances a pole, or walks — that moment of "it actually works, and I didn't tell it how" is the payoff. Chase it deliberately: form a hypothesis before you run an experiment, and notice when you're surprised. That's usually where the real insight is.

## The arc, project by project

Structurally, the term splits into two movements. **Projects 1–3 build the pieces in isolation** — a body with no brain, an optimizer with no body, a network with no body either — so you understand each one on its own before they're combined. **Projects 4–6 then fuse all three into one pipeline and hold that pipeline fixed while the body gets harder**: a vehicle, then an unstable pole-balancer, then a six-legged walker — with a second controller architecture and a second evolutionary algorithm folded in along the way, so the last project is also where the most ideas are simultaneously in play.

| # | Project | What's new |
|---|---|---|
| 1 | **Braitenberg Vehicles** | Embodied behavior with *zero* learning — hand-wired reflexes that still look intelligent |
| 2 | **Evolutionary Algorithms** | The optimization machinery itself: selection, mutation, elitism, GA vs. ES |
| 3 | **Neuroevolution (XOR)** | Neural networks as evolvable genomes, no gradients required |
| 4 | **Embodied NeuroEvolution I: Braitenberg Phototaxis** | Put an evolved network in the Project 1 robot's body — hand-wiring vs. evolution, head to head |
| 5 | **Embodied NeuroEvolution II: CartPole Balancing** | Same pipeline, a harder, unstable dynamical task: CartPole |
| 6 | **Embodied NeuroEvolution III: Six-Legged Walker** | Same pipeline, a much bigger body, a second controller architecture (CTRNN) with real memory, a second algorithm (CMA-ES) — and, optionally, a real-physics sim-to-real test |

**1 — Braitenberg Vehicles: Phototaxis.** A two-wheeled vehicle with two light sensors, wired directly to two motors. No planning, no memory, no network — and yet crossing the wires one way produces light-seeking behavior, and the other way produces something else entirely. This is the thesis statement for the whole course: *complex-looking behavior doesn't require a complex controller.*

**2 — Evolutionary Algorithms.** Strip away the robot and study the optimizer itself. You'll run a Genetic Algorithm and an Evolution Strategy on a simple genome and see, directly, what population size, mutation rate, and selection pressure actually do to a search process — the mechanics you'll rely on, mostly without re-deriving them, for the rest of the term.

**3 — Neuroevolution on XOR.** Put a neural network's weights into a flat vector and evolve them instead of backpropagating. XOR is a deliberately small, well-understood task — small enough that you can reason precisely about *why* a network needs a hidden layer, and *why* evolution can find good weights without a gradient.

**4 — Embodied NeuroEvolution I: Braitenberg Phototaxis.** The Project 1 vehicle, with its hard-wired reflex replaced by an evolved neural controller. You'll compare the evolved controller against your own hand-designed one — same body, same task, two very different design processes — and watch some behaviors emerge that hand-wiring literally couldn't produce.

**5 — Embodied NeuroEvolution II: CartPole Balancing.** Same neuroevolution pipeline, a much less forgiving task: an inherently unstable system that fails within seconds without control. This is your second pass at the full evolve → simulate → analyze workflow, and a bridge to a standard RL benchmark you'll see again elsewhere.

**6 — Embodied NeuroEvolution III: Six-Legged Walker.** The same evolve → simulate → analyze pipeline, on a six-legged body that only moves forward if the legs currently in stance form a stable tripod — so evolution's first job is discovering *any* coordinated leg-timing pattern before it can be shaped into an efficient gait. You'll evolve two controller architectures on the identical task and fitness signal: a feedforward network with no memory at all, and a CTRNN (continuous-time recurrent neural network) whose neurons carry their own internal state and can sustain a walking rhythm with zero sensory input, the way a real insect's central pattern generator does. An optional, extra-credit part then asks a harder question than "does evolution converge": does a gait evolved on this project's idealized physics still walk when you drop it onto a real-contact-dynamics MuJoCo simulation it never saw during evolution?

## What carries across every project

- **A fitness function is a hypothesis about what "good" means.** You'll write several over the course of the term, and the projects deliberately ask you to reflect on what each one rewards — and what it accidentally rewards instead.
- **Evolution is stochastic.** A single run tells you much less than you'd think. Use `--seed` when you want a specific run to be reproducible, but run multiple seeds before you trust a conclusion about *typical* behavior.
- **Change one thing at a time.** Projects 1–3 give you a `study.py` sweep-and-plot script for exactly this reason — isolate a variable, hold everything else fixed, and let the plot make the argument. From Project 4 on, you're expected to write that sweep-and-plot code yourself (Project 6 also ships two sweep scripts, `hexapod_torque_sweep.py`/`hexapod_timescale_sweep.py`, but they're specific to its optional sim-to-real test, not a general-purpose replacement for `study.py`) — by then the pattern should be familiar enough to build without a template.
- **A prediction beats a post-hoc explanation.** Several projects explicitly ask you to write down what you expect *before* you run the experiment. Do this even when it isn't asked — being surprised by your own results is one of the fastest ways to actually learn something here.
- **The pipeline is the point, not just the result.** Projects 4–6 reuse the same evolve → simulate → analyze structure on three different bodies, so that by Project 6 the workflow itself is second nature and your attention goes to what's actually new: a controller with real memory, a second evolutionary algorithm, a much larger genome, and — if you attempt the optional part — a real physics engine that doesn't care what your idealized model assumed.

## How each project is graded

Every project is worth **10 points**, split the same way throughout the term:

- **5 pts** — completion of each assignment part (roughly equal weight per part).
- **1 pt** — a complete title page (name, course, assignment name, date, time spent, self-assessment).
- **2 pts** — figures: readable, labeled, meaningful, and paired with interpretation in the text. A plot with no discussion (or discussion with no plot) won't get full credit.
- **2 pts** — creativity, insight, and critical thinking beyond the minimum needed to answer each question.

That last category matters more than it might look on paper. The guiding questions in each project are a floor, not a ceiling — going one step further with your own question is exactly what it's there to reward.

One exception to the 10-point pattern: Project 6 offers up to **1 additional point of extra credit**, on top of its normal 10, for a genuine attempt at its optional sim-to-real transfer test. See that project's own README for what counts as a genuine attempt.

## Before you start: setup

Each project is self-contained: it has its own `requirements.txt` and its own virtual environment, created from inside that project's folder. There's no single shared environment for the whole course — set one up fresh the first time you start each project, following the "Installation" section of that project's own README.

1. **Start with Project 1.** From inside `1_Braitenberg/`, follow its README's Installation section to create a virtual environment and install `requirements.txt`. Then confirm your environment works:

   ```bash
   cd 1_Braitenberg
   python sim.py --viztraces
   ```

   This will raise a `NotImplementedError` until you complete Part 1 of that project (`Vehicle.think()` is intentionally left unimplemented) — that's expected, not a setup problem. If you instead get an import error, or no plot window appears at all, check the notes in that project's README before troubleshooting further.

2. **Read each project's README before touching its code**, and set up its environment the same way before starting. They're not boilerplate — Background sections explain concepts you're expected to already understand before Part 1, and the Tips section up front exists specifically to save you time.

## General guidelines that apply to every report

- Submit a single PDF per project to Moodle, with the title page described above.
- Organize your report by assignment part, and treat each part as *figure + written discussion*, not one or the other.
- Prioritize insight over volume — a focused paragraph beats a page of restated code output or console logs.
- Use `--seed` liberally while debugging, and drop it (or run multiple seeds) when you want to report on typical behavior rather than one lucky (or unlucky) run.

## Contact

This collection of projects was developed by Eduardo Izquierdo for **ECE497 (Fall 2026): Evolutionary Robotics** at Rose-Hulman Institute of Technology.
