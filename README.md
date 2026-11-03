# BrahClaude

**Does a homeostatic pinger come to know the thing it pings, and what changes when the thing being pinged is itself alive (tiring, rewiring)?**

This grows out of the old PerceptionLab `ecg.json` loop (see [GeometricNeuronOriginReview](https://github.com/anttiluode/GeometricNeuronOriginReview)). There, a Homeostatic Coupler sets a checkerboard's square size, four resized pixels come back, and the coupler regulates their variance. The question raised on 3 Nov 2026:

> If the coupler were a neuron and the checkerboard were the neurons in between, it would learn to know them through its pings, with its goal being to stay in homeostasis. And what is the difference between the neurons in between and the chessboard?

And Sol's answer, which this repo turns into code:

> The chessboard is a fixed transformation; the neurons in between are an adaptive, history-dependent medium that can themselves learn from the interaction.

Sister repo: **BrahSol** (Sol / ChatGPT).

Run it: `pip install numpy opencv-python`, then `python run_gates.py` (about 3 min) and `python gate2b_coupler_paired.py` (about 2 min). Results go to `results/`.

---

## The pieces

| file | what it is |
|---|---|
| `brah/ecg_loop.py` | Qt-free port of the original five nodes. Node bodies are copied; only the scheduler is reconstructed. |
| `brah/media.py` | What sits where the checkerboard sat: **Checkerboard** (fixed geometry), **static-net**, **habituating** and **plastic**. |
| `brah/pinger.py` | The pinger: a memory table plus five policies for choosing the next ping. |

**The media.** A 48-neuron rate network. Input neurons have Gaussian tuning curves over the ping value u ∈ [0, 1]. The four "taps" get no direct input and only hear the network through fixed random mixed-sign weights, so the return is never just a tuning curve. The three variants:

- **static-net:** that network as described.
- **habituating:** adds fatigue: a neuron that fires builds adaptation (τ = 20 ticks) and recovers when rested.
- **plastic:** adds map plasticity on top of fatigue: each input neuron's preferred ping drifts toward the pings that make it fire, so frequently used pings recruit more neurons.

**The pinger's knowledge** is a table with one entry per ping (32 levels). Each entry holds an exponentially forgetting estimate of the 4-tap return, the recent prediction error, and the time it was last asked. It is scored against the medium's true *current* response map, which is measured on clones so that measuring doesn't disturb the run. That clone access is a privilege no real neuron has, and it is used only for scoring.

**The policies:**

- `coupler`: the original edge-of-chaos Homeostatic Coupler, unchanged. It never reads the table; the table fills passively from whatever it happens to ping.
- `random`, `sweep`: baselines.
- `curious`: asks where the expected error is largest, scored as recent error × time since last asked.
- `dual`: Sol's J(u), homeostasis minus λ·curiosity. It uses its own table to predict what each ping would do to the return variance.

---

## Gate 0: the original loop, reproduced

| setup (sequential schedule) | period | square sizes visited |
|---|---|---|
| dim 256, window 50 | **52 ticks** | 8 |
| dim 256, window 30 | **32 ticks** | 8 |
| dim 1024, window 50 or 30 | 2 (short alternation) | 2 |
| dim 2048, window 50 | 51 | 3 |

This matches the review README's reconstruction receipt: about 52 ticks at window 50, about 32 at window 30, and the 1024 short regime. A synchronous schedule gives 55/34/4.

Sol's static distinguishability counts also reproduce exactly. Out of 55 square sizes, the distinct 4-tap vectors are **55 / 28 / 19** at dims 256 / 1024 / 2048. Once the coupler sums the four taps (`get_blended_input('signal_in', 'sum')`), the distinct values are **38 / 20 / 17**.

**What Gate 0 adds:** the ECG pulse is a limit cycle over **8 of the 55** possible square sizes. The original coupler is not exploring the chessboard; it is circling a small patch of it.

## Gates 1–3: who comes to know the medium?

All results are 10 seeds × 2,000 ticks. **err** is the late-run mean absolute error of the pinger's table against the true current map; lower means it knows the medium better. **homeo** is the mean |var(return, 50 ticks) − 0.1|; lower means better homeostasis. **cov** is the number of the 32 pings ever used.

| medium | coupler err / homeo / cov | random | sweep | curious | **dual** |
|---|---|---|---|---|---|
| checkerboard* | .231 / **.009** / 9 | 0 / .718 | 0 / .739 | 0 / .764 | 0 / .070 |
| static-net | .205 / .203 / 9.3 | .072 / .284 | .092 / .272 | .066 / .262 | .068 / **.132** |
| habituating | .198 / .319 / 13.9 | .059 / .270 | .080 / .275 | .054 / .231 | .054 / **.129** |
| plastic | .190 / .330 / 19.9 | .051 / .310 | .073 / .294 | .046 / .303 | .049 / **.140** |

\* The checkerboard is deterministic, so one visit per ping makes the table exact. In Gates 1–3 its ping is clipped to [0, 1].

### Gate 2B: the paired test (60 seeds, same wiring across the three media)

| medium | coupler coverage (mean / median) | seeds locked on ≤ 4 pings | paired gain vs static | coupler err | coupler homeo |
|---|---|---|---|---|---|
| static-net | 8.2 / 4 | **53%** | — | .215 | .137 |
| habituating | 16.3 / 19 | 13% | **+8.1 ± 1.3** (41 seeds up, 11 down) | .176 | .230 |
| plastic | 22.6 / 22 | **2%** | **+14.4 ± 1.1** (54 up, 4 down) | .175 | .280 |

## Verdicts

1. **The intuition is half right, and measurably so.** An adaptive medium does break the coupler out of its limit cycle. Against a static network it locks onto at most 4 pings in about half the seeds. Against a tiring network it explores about twice as many pings, and against a rewiring network it almost never locks. Nothing in the coupler changed; the world refusing to hold still did it.

2. **But the coupler explores because it is failing, not because it is curious.** As coverage rises, its homeostasis gets worse (.137 → .230 → .280). Its knowledge barely improves (err .215 → .175), against .05 for anything that explores on purpose. This is Ashby's *ultrastability* (Design for a Brain, 1952): when the regulated variable leaves its bounds, behaviour changes until something holds. The search is blind, and nothing it finds is kept in the policy.

3. **Sol's dual objective gets both.** On every living medium, `dual` knows the medium as well as pure curiosity does (.054 vs .054 habituating, .049 vs .046 plastic). It also holds homeostasis about 2× better than the original coupler and better than every other policy. The missing ingredient was memory that the policy actually reads: the table changes the next ping. Prior art here is dual control (Feldbaum, 1960–61), active inference / free energy (Friston), and curiosity as learning progress (Schmidhuber 1991; Oudeyer 2007). This is a working instance of those ideas, not a new one.

4. **Back-action: a narrow pinger rewires the medium more.** In the plastic medium, the coupler moved the network's tuning further (drift .558) than any broad policy (.458–.463), because the network specialises to its few habitual pings. I tried to measure "the medium learned the pinger" directly, as the correlation between ping usage and gain change. It came out weak (r ≤ 0.30) and isn't claimed.

5. **The checkerboard is the one world where the original coupler is excellent** (homeo .009). It was tuned to that world. That is also the answer to "what's the difference": a fixed geometry rewards a fixed habit, and a living medium punishes it.

## What failed, kept on purpose

- Hebbian plasticity on the recurrent weights (Oja, then BCM at η = 0.003–0.03) collapsed the network into an input-blind attractor every time: the response spread across pings went to 0. A medium that learns that way goes deaf, and a deaf medium is trivially "known". So plasticity was moved to the tuning curves (map expansion, as in Recanzone et al. 1993).
- The coupler's parameters (target variance 0.1, setpoint 0.6) were tuned for the checkerboard and never retuned for the networks. Some of its homeostasis gap on networks may be mistuning. A retuned-coupler attacker is listed below.

## Limits

These are designed toy worlds with parameters I chose (tap weight scale, fatigue τ, plasticity η). The memory table cannot generalise to pings it has never tried; it only remembers. Scoring by cloning the medium is an oracle that real neurons don't have.

## Next gates (open)

- **G4 – "ping what it wants".** Give the pinger a target return pattern and count pings to reach it, in a static vs a plastic medium. This is the inverse model, the "learn to ping what it wants" half of the idea.
- **G5 – the neurons in between are pingers too.** Two coupled learners, each with its own homeostatic goal. Who comes to know whom, and does the channel between them get richer or collapse?
- **G6 – Sol's generalisation test.** A smooth forward model scored on held-out pings it never tried, so "knowing" means predicting, not remembering.
- **Attacker: a retuned coupler** (grid search over its setpoint and target variance on each medium) before claiming verdict 2 is about homeostasis rather than tuning.
