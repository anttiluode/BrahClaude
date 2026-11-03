# BrahClaude / pinger: the homeostatic pinger and the neurons in between

Part 1 of the repo (morning of 3 Nov 2026). Part 2, the temporal-Sihti routes, is in the [root README](../README.md).

**Does a homeostatic pinger come to know the thing it pings come to know the thing it pings, and what changes when the thing being pinged is itself alive (tiring, rewiring)?**

This grows out of the old PerceptionLab `ecg.json` loop (see [GeometricNeuronOriginReview](https://github.com/anttiluode/GeometricNeuronOriginReview)). There, a Homeostatic Coupler sets a checkerboard's square size, four resized pixels come back, and the coupler regulates their variance. The question raised on 3 Nov 2026:

> If the coupler were a neuron and the checkerboard were the neurons in between, it would learn to know them through its pings, with its goal being to stay in homeostasis. And what is the difference between the neurons in between and the chessboard?

And Sol's answer, which this repo turns into code:

> The chessboard is a fixed transformation; the neurons in between are an adaptive, history-dependent medium that can themselves learn from the interaction.

Sister repo: **BrahSol** (Sol / ChatGPT).

Run it from the repo root (`pip install -r requirements.txt`):

- `python pinger/run_gates.py` (about 3 min)
- `python pinger/gate2b_coupler_paired.py` (about 2 min)
- `python pinger/attacker_retuned_coupler.py` (about 4 min)
- `python pinger/gate4_ping_what_it_wants.py` (about 6 min)
- `python pinger/gate4b_why_easier.py` (about 3 min)

Results go to `pinger/results/`. The library code lives in `brah/` beside the temporal-Sihti `edges.py`.

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

3. **Sol's dual objective gets both.** On every living medium, `dual` knows the medium as well as pure curiosity does (.054 vs .054 habituating, .049 vs .046 plastic). It also holds homeostasis about 2× better than the original coupler and better than every other policy. ⚠️ **Corrected by the retuned-coupler attacker below:** the 2× homeostasis edge over the coupler was mostly mistuning. A coupler retuned per medium regulates *better* than `dual` on the static and tiring networks. `dual` is the only policy that does well at both, but it does not beat a specialist at regulation. The missing ingredient was memory that the policy actually reads: the table changes the next ping. Prior art here is dual control (Feldbaum, 1960–61), active inference / free energy (Friston), and curiosity as learning progress (Schmidhuber 1991; Oudeyer 2007). This is a working instance of those ideas, not a new one.

4. **Back-action: a narrow pinger rewires the medium more.** In the plastic medium, the coupler moved the network's tuning further (drift .558) than any broad policy (.458–.463), because the network specialises to its few habitual pings. I tried to measure "the medium learned the pinger" directly, as the correlation between ping usage and gain change. It came out weak (r ≤ 0.30) and isn't claimed.

5. **The checkerboard is the one world where the original coupler is excellent** (homeo .009). It was tuned to that world. That is also the answer to "what's the difference": a fixed geometry rewards a fixed habit, and a living medium punishes it.

## What failed, kept on purpose

- Hebbian plasticity on the recurrent weights (Oja, then BCM at η = 0.003–0.03) collapsed the network into an input-blind attractor every time: the response spread across pings went to 0. A medium that learns that way goes deaf, and a deaf medium is trivially "known". So plasticity was moved to the tuning curves (map expansion, as in Recanzone et al. 1993).
- The coupler's parameters (target variance 0.1, setpoint 0.6) were tuned for the checkerboard and never retuned for the networks. Some of its homeostasis gap on networks may be mistuning. A retuned-coupler attacker is listed below.

## Attacker: the retuned coupler (`pinger/attacker_retuned_coupler.py`)

Was the original coupler only losing on the networks because its knobs were set for the checkerboard? I grid-searched setpoint × internal target variance × sharpness (4 × 4 × 3 = 48 settings) per medium on tuning seeds 100–104. The winner was then scored on held-out seeds 0–19. The homeostasis goal is the same fixed |var − 0.1| as everywhere.

| medium | default coupler homeo / err / cov | **retuned coupler** homeo / err / cov | dual homeo / err / cov |
|---|---|---|---|
| static-net | .178 / .212 / 8.6 | **.061** / .221 / 5.7 | .122 / .074 / 32 |
| habituating | .223 / .173 / 15.5 | **.074** / .187 / 11.8 | .126 / .048 / 32 |
| plastic | .229 / .168 / 22.1 | .172 / .187 / 20.1 | **.140** / .050 / 32 |

- **Verdict 3 corrected.** Tuned, the coupler is the better pure regulator on fixed and tiring media. It loses only on the rewiring medium, where its tuned setting didn't hold up: .079 on the tuning seeds, .172 held out, because the medium it was tuned on keeps changing. `dual` is the only policy that does well at both regulation and knowledge, but it does not dominate.
- **Verdict 2 strengthened.** The better tuned the coupler, the *less* it explores (5.7 / 11.8 / 20.1 pings) and the less it knows (err .19–.22, against .05–.07 for `dual`). Good regulation and getting to know the medium trade off for a coupler, because its exploration only happens when regulation fails.

## Gate 4: "ping what it wants" (`pinger/gate4_ping_what_it_wants.py`)

Each episode the pinger gets a target return y\* (the medium's current answer to a random ping, so it is reachable when set). It must produce a return within 0.15 of y\*, with at most 32 pings. The medium persists across 300 episodes, over 20 seeds.

- `blind`: random pings, no repeats within an episode.
- `ashby`: random jump on every miss, repeats allowed. This is Ashby's homeostat used as a goal-seeker.
- `memory`: try pings in order of how close their *remembered* return is to y\*. It re-ranks after every miss, and the same Gate 1–3 table persists across episodes.
- `oracle`: the same ordering read off the true current map; the floor.

| medium | blind | ashby | **memory** (first 20 → last 100 episodes) | oracle |
|---|---|---|---|---|
| checkerboard | 11.0 | 13.9 | 3.04 → **1.00** | 1.00 |
| static-net | 15.9 | 16.8 | 9.50 → **7.62 ± .67** | 1.00 |
| habituating | 15.6 | 16.6 | 8.52 → **6.25 ± .46** | 1.00 |
| plastic | 12.7 (was 15.9) | 13.4 (was 16.2) | 9.03 → **6.17 ± .50** | 1.00 |

Values are mean pings to hit a target over the last 100 episodes, with failures counted as 32; ± is the standard error over seeds. Success within 32 pings, last 100 episodes: blind .71–.81, ashby .67–.76, memory .80–.84 on the networks; everything is 1.00 on the checkerboard except ashby (.86).

**Verdicts**

1. **Memory roughly halves the pings needed to get what it wants** (about 6–8 against about 16 on every network), and it keeps improving over episodes. On the deterministic checkerboard it becomes perfect: one ping per target after about 50 episodes. This is "learn to ping what it wants": an inverse model built from the forward memory.
2. **Ashby's homeostat is no better than blind search** (slightly worse, because it repeats pings). Ultrastability finds *a* stable setting; it doesn't find *the wanted* one.
3. **A living medium did not hurt memory.** If anything the static network was the hardest (7.6 vs 6.2). The likely reason is that fatigue damps the network's persistent activity, so its answers depend less on recent history. That explanation is **untested**.
4. **The rewiring medium becomes easier to please, even for a pinger that learns nothing.** Blind search speeds up from 15.2 to 12.9 pings across the episodes, and its success rate rises from .71 to .81, only in the plastic medium; static and tiring stay flat. Gate 4B (`pinger/gate4b_why_easier.py`, rested maps, 20 seeds) shows why. After 300 goal episodes, the number of pings that can hit any given target rises from 3.46 to **4.24** (blind) or **4.28** (memory). The number of distinct responses falls from 17.4 to **16.5**, and the spread across pings from 1.148 to 1.12–1.13. **The medium gets easier to drive by losing distinctions.** That is the "neurons in between learn the pinger" effect, measured: it adapts toward being easy to drive, which costs it resolution. It is modest (+23% hit-redundancy) and specific to this plasticity rule.

## Limits

These are designed toy worlds with parameters I chose (tap weight scale, fatigue τ, plasticity η). The memory table cannot generalise to pings it has never tried; it only remembers. Scoring by cloning the medium is an oracle that real neurons don't have.

## Next gates (open)

- ~~G4 – "ping what it wants"~~: done, see above.
- **G4C – does the lost resolution matter?** Pair goal-pinging with a *discrimination* task, and check whether a medium that became easy to please can still tell targets apart that it used to.
- **G5 – the neurons in between are pingers too.** Two coupled learners, each with its own homeostatic goal. Who comes to know whom, and does the channel between them get richer or collapse?
- **G6 – Sol's generalisation test.** A smooth forward model scored on held-out pings it never tried, so "knowing" means predicting, not remembering.
- ~~Attacker: a retuned coupler~~: done; it corrected verdict 3.
