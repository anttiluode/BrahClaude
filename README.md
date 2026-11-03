# BrahClaude — routes that keep a Sihti residue of their own pulses

**Question (Nov 3 2026):** what if the pulse travelling along a route carries the
route's *residue* of earlier pulses, Sihti-style, and each new pulse adds to it?
Then a neuron pinging the network reads (and writes) the hidden state *on the
routes*, not just on the neurons.

> **Two parts.** This page is part 2, the temporal-Sihti routes (afternoon of 3 Nov). Part 1, the
> homeostatic pinger and the "neurons in between" (gates 0–4, morning), is in [pinger/](pinger/README.md).

This repo builds the smallest version of that and tests it with gates that were
pre-registered before any run ([PREREGISTRATION.md](PREREGISTRATION.md), commit
`1bcba05`). Plain numpy, CPU, ~35 s: `python run_all.py` (G0–G5 and the D1/D2 diagnostics). G6 takes about 35 min: `python experiments/g6_one_soma.py`. Tests: `pytest tests`.

## The edge

Each route keeps K=8 exponential traces of the pulses that crossed it,
`m_k(t+1) = λ_k m_k(t) + x(t)`, `τ_k = 2, 4, …, 256` steps. Its Sihti residue
stack is the octave-age bins of its own history,

```
h_0 = m_0,   h_k = m_k − m_{k−1}      Σ_k h_k = m_{K−1}   (exact telescoping)
```

and a pulse leaving the route is

```
y(t) = x(t) · max(0, 1 + g·h(t))
```

The spike itself is stereotyped (0/1). What it *carries out* is decided by the
route's band gains `g` and the residue earlier pulses left there. Unclipped, this is
exactly a second-order Volterra system,
`y = x(t)[1 + Σ_s κ(s) x(t−s)]`, `κ(s) = Σ_k (g_k − g_{k+1}) λ_k^{s−1}`, so the band
basis is a sum-of-exponentials kernel written as an octave residue stack.

![kernels](figures/kernels.png)

Four route types: **FAC** (facilitating), **DEP** (depressing), **BAND** (depress
then facilitate, peak at age 13), **AUG** (slow augmentation, peak at age 52).
These are rough stand-ins for real short-term plasticity at its separated
timescales (facilitation, depression, augmentation, post-tetanic potentiation).

## Results (as pre-registered)

| Gate | What | Result |
|---|---|---|
| G0a | telescoping Σh = m_{K−1} | **PASS** (err 3.6e-15) |
| G0b | edge == explicit Volterra kernel | **PASS** (err 5.3e-15) |
| G0c | first pulse on a fresh route returns exactly 1 for every type | **PASS** — reading a route requires a prior write |
| G1 | identify hidden route type by probing (6 pulses, σ=0.3) | **FAIL** — optimised 0.994 vs best fixed (burst) 0.947: +0.047, threshold was +0.05 |
| G2a | silent memory read by a neutral ping | **PASS** — 0.996 accuracy; delay output exactly 0; reset control 0.125 (chance) |
| G2b | amount vs age from one ping | **PASS** — octave-tuned terminals 0.85, homogeneous 0.41 |
| G3a | reward-only learner writes on the right route | **PASS** — FAC: 100 % on target; DEP: 100 % on the *other* route |
| G3b | learned write timing tracks the residue peak | **PASS** — BAND lead 12.5 at W=4 (κ peak 13), falls to 4.5 at W=128 |
| G3c | learner ≥ 95 % of greedy | **PASS** — all six cases within 0.1 % |
| G4 | two messages down one axon, separated by the routes | **FAIL** — corr(â, a) 0.757 < 0.8; every other criterion passed |
| G5-0a | linear route: a neutral read is a common-mode write | **PASS** (err 1.8e-15) |
| G5-0b | fresh Tsodyks–Markram route answers exactly 1 | **FAIL as written** — off by 2.2e-16 (float rounding); holds to machine precision |
| G5-1 | linear route: reading is free (cost ≤ 0.02) | **PASS** — cost −0.001 / +0.001 / +0.007 |
| G5-2 | TM route: dense reads destroy the memory (cost ≥ 0.30) | **PASS** — cost 0.866 at period 2 |
| G5-3 | TM: read cost shrinks with spacing | **PASS** — 0.866 → 0.302 |
| G5-4 | TM: selective re-firing preserves it | **PASS** — 1.000 / 0.967 / 0.976 vs 0.989 / 0.945 / 0.432 unread |
| G6-0 | one soma, broadcast pings: zero information (exact) | **PASS** — posterior = prior to 7.6e-17 |
| G6-1 | linear route: choosing which routes to ping recovers the pair through one soma | **PASS** — 0.992 by query 6, 1.000 by 12 (random subsets 0.887 at 6) |
| G6-2 | TM: destructive reads cap what one soma can learn | **PASS** — 0.553 vs 1.000 linear |
| G6-3 | the questioning costs TM memory, not linear | **PASS** — TM 0.832 → 0.043; linear 0.684 → 0.687 |

### G1 — probing a route (fail, narrowly)

The learner's train is `[8, 9, 10, 11, 12, 20]`: a short burst (separates FAC from
DEP from the rest) and then one pulse ~8–12 steps later, aimed at the age where
BAND's residue peaks and AUG's has barely started. Error rate falls from 5.3 % (best
fixed train) to 0.6 %, a ninefold reduction, but the accuracy gain is +0.047 and I
pre-registered +0.05. Fail as written; the margin was a ceiling effect I did not
anticipate (the burst train is already strong).

### G2 — silent memory, and why the octave stack matters

G2a is the Wolff et al. (2017) "ping" shape: a burst written by someone else on 2 of
8 routes, 150 silent steps (output exactly zero; nothing for a passive observer to
see), then one neutral pulse on all routes reads which routes were written with
99.6 % accuracy. Resetting the residues to their mean kills it (12.5 %, chance).

G2b is the Sihti-specific part. One band projection cannot tell "4 pulses long ago"
from "2 pulses recently":

![amount vs age](figures/g2b_amount_vs_age.png)

Four terminals of one axon tuned to different octave bands read different projections
of the residue stack, and one ping then decodes both **how much** and **when** (16
classes, 0.85 vs 0.41).

### G3 — writing residue for a later pulse

A pinger places 6 pulses before t0; a third party's spike then arrives on both routes
somewhere in a window W. The learner is rewarded only for what that later spike
delivers. It finds, without a model:

- FAC routes: prime the target route. DEP routes: prime the *other* route to block it.
- Timing: lead ≈ κ-peak − W/2 (BAND: 12.5 → 6.5 → 4.5; AUG at W=32: 37.7 vs 52 − 16 = 36).

![write for later](figures/g3_write_for_later.png)

**Honest caveat:** because the history is linear before the clip, the optimal policy
is a matched filter (put pulses at the top slots of E[κ] on the sign-matched route),
and the learner simply finds that. It shows that reward-only learning discovers
write-for-later; it does not show anything a matched filter would not.

### G4 — burst multiplexing (fail on a threshold above the noise ceiling)

One spike train carries two independent slow signals: event rate a(t) and burst
probability b(t). A depressing target and a facilitating target on the same axon
separate them: the depressing readout tracks the event rate, and
facilitating/depressing tracks burstiness (corr 0.82). Cross-talk −0.07 and −0.02.
A residue-free route mixes both (corr 0.64 with a, 0.40 with b), and with flat routes
the ratio is identically 1, carrying no b-information at all.

![multiplex](figures/g4_multiplex.png)

corr(â, a) is 0.757, below the pre-registered 0.8. The diagnostic shows why: an ideal
counter of the true event onsets reaches only 0.757 against a(t) (Poisson shot noise
at this readout timescale), and the depressing route matches that ideal counter at
0.9996. The route did its job; my threshold sat above the information ceiling. FAIL
as written.

Two bugs were fixed after G4's first run, both disclosed: the low-pass filter ran
along the wrong array axis (all correlations ≈ 0), and the flat control used
`F/(D+1e-6)`, which made an identical-route ratio spuriously non-constant.

### G5 — is a ping read destructive? (pre-registered after G0–G4, commit `bc677bb`)

In the linear route above, a neutral ping writes the same pulse on every route, so it
shifts all their answers equally and the differences holding the memory survive any
number of reads. Real short-term plasticity is multiplicative and saturating, so G5
reruns the G2a task (burst on 2 of 8 routes, then silence) with 12 reads at period P.
It does this on the linear AUG route and on a Tsodyks–Markram route (`brah/tm.py`,
Mongillo et al. 2008 proportions). **Read cost** = accuracy of one unread look at the
time of the 12th read, minus accuracy of the 12th of 12 neutral reads.

| route, period | 1st read | 4th | 12th | one unread look at the same time | selective re-firing (oracle), then read |
|---|---|---|---|---|---|
| linear, P=2 | .993 | .991 | .985 | .984 | 1.000 |
| linear, P=10 | .993 | .980 | .876 | .876 | 1.000 |
| linear, P=40 | .993 | .859 | .287 | .294 | 1.000 |
| **TM, P=2** | .993 | **.467** | **.123** | **.989** | 1.000 |
| **TM, P=10** | .991 | **.257** | **.128** | **.945** | .967 |
| **TM, P=40** | .993 | .294 | .130 | .432 | .976 |

Chance is .125. Noise σ is 0.25 × each model's first-read gap.

- **On the linear route, reading is free.** The 12th read equals an unread look at the same moment; only time erases the memory. That's the exact common-mode identity G5-0a.
- **On the TM route, you get one good look.** The first read is .99. By the 4th read at P=2 it is .47, and by the 12th it is chance, while the same memory left unread is still .99. Each neutral ping pushes every route toward the same facilitated, depleted state, which erases the differences. Spacing the reads helps (cost .87 → .30), because the routes recover between looks.
- **Re-firing only the written routes keeps it** (Mongillo's refresh), but only if the reader already knows what was written. That is an efference copy. At P=2 the code *flips sign*: the refreshed routes are resource-depleted, so they answer *below* the unwritten ones (gap −0.86). A reader assuming "bigger = written" would decode exactly backwards; the template decoder survives only because it is given the dynamics.
- **What this means for the repo:** G2a's 99.6% silent-memory read holds for **one** read. "A neuron pinging its network to learn it" pays for each look, and on real synapses the bill is the memory itself. This is the same back-action as `pinger/` Gate 4B, where probing a plastic medium made it lose distinctions.

Disclosed:
- An informal pilot with other settings was run in chat before the pre-registration (its numbers are in PREREGISTRATION.md).
- The first G5 run had a scoring bug: arm 0 was decoded at read index 0, where that arm has no pulse, giving chance. The pre-registration says t_11. It was fixed and rerun; no model or criterion changed.

### G6 — addresses through one soma, when every look costs memory (pre-registered, commit `83a50a4`)

Sol (BrahSol) spotted that G2a's 99.6% needs **eight separately recorded routes**.
Summed into one soma, the way the original coupler summed its four taps, every item
gives the same total, so the memory is invisible. G6 makes that the task. There are 8
routes, and the hidden state is which of all **28** route pairs was written. The prober
pings a **subset** of routes and sees **one summed number**. It decodes with exact
Bayes, given the dynamics and its own history (both are gifts, and stated as such).
1000 episodes per cell, 12 queries.

| P = 10 | broadcast | random half | greedy info-gain | 8 separate channels | memory left after (unqueried) |
|---|---|---|---|---|---|
| linear, query 6 / 12 | .031 / .031 | .887 / .996 | **.992 / 1.000** | 1.000 / 1.000 | .687 (.684) |
| TM, query 6 / 12 | .039 / .039 | .474 / .544 | **.532 / .553** | .978 / .978 | **.043** (.832) |

| P = 40 | broadcast | random half | greedy | 8 channels | memory left (unqueried) |
|---|---|---|---|---|---|
| linear, query 12 | .036 | .866 | **.949** | 1.000 | .103 (.102) |
| TM, query 12 | .032 | .649 | **.808** | .992 | .060 (.177) |

Chance is 1/28 = .036.

1. **One soma plus broadcast pings is blind, exactly.** The posterior never moves from the prior (7.6e-17).
2. **An address can be a sequence of interventions.** On the linear route, choosing *which* routes to ping recovers the written pair through one summed readout. Greedy info-gain gets there by query 6 with about 3.8 routes per query, and the questioning costs nothing.
3. **On a realistic synapse the address costs the content.** Each subset ping erases what it asks about, so one soma tops out at .55 (P = 10). Eight separate channels get .97 in a single look. Spacing helps (.81 at P = 40) because routes recover between looks. Any questioning wipes the TM memory for a later reader (.832 → .03–.05), including the 8-channel look.
4. **Greedy info-gain is damage-blind.** At P = 10 on TM it barely beats random subsets (.553 vs .544). It maximises the information in the *next* answer, not what is left for later. A damage-aware controller is the open problem, and it is exactly Sol's "learned conservation" test in BrahSol.

Consequence for the neuron picture: if synapses are destructive like this, a single-compartment reader is badly placed. Separate readout compartments (dendritic branches as partly independent units, the biology Sol cited) do in one look what one soma cannot finish at all. That is a model result, not a biological claim.

### Post-hoc diagnostics D1, D2 (requested by Sol; declared in PREREGISTRATION.md, not pre-registered)

- **D1, G2b with a matched raw-trace baseline:** octave-band terminals .857, four terminals each reading one raw trace (scaled to the same size) .760, homogeneous .408. The bands carry no new information (they are an invertible transform of the traces). But band-pass terminals, each reading a *difference* of traces, are a better default projection than low-pass single traces. That is the accurate version of "Sihti helps".
- **D2, is G5's erasure real or only in the readout?** On the TM route, the written-minus-unwritten facilitation gap Δu just before the 12th read is .019 after 11 neutral reads, against .218 unread (P = 2). At P = 10 it is .014 vs .163, and at P = 40 .005 vs .054. The memory is gone from the synapse state itself.

## Prior art (none of the mechanisms here is new)

- Short-term plasticity as edge state: Tsodyks & Markram 1997; Abbott et al. 1997.
- The probe Δr = r_B − r_A is the **paired-pulse ratio**, a standard electrophysiology tool.
- Silent synaptic working memory, refreshed by selective re-firing: Mongillo, Barak & Tsodyks 2008.
- Pinging to read hidden state: Wolff, Jochim, Akyürek & Stokes 2017.
- Burst multiplexing read by facilitating vs depressing synapses: Naud & Sprekeler 2018.
- Bank of exponentials as a history code: Shankar & Howard 2012 (Laplace-transform time cells).
- Choosing subset queries through one summed readout is group testing (Dorfman 1943) with noisy, *destructive* tests; the greedy info-gain policy is Bayesian experimental design (Lindley 1956).
- Dendritic branches as partly independent computational compartments: Poirazi, Brannon & Mel 2003; London & Häusser 2005.

What this repo adds is small and specific: the route history written as an exact Sihti
residue stack (octave age bins), and the measurement that octave-tuned terminals let
one ping separate **amount** from **age** (G2b) where a single projection cannot.

## Open

- **Non-destructive reading on saturating routes:** which ping patterns (spacing, strength, only some routes) recover the most information per unit of memory destroyed? This is the price-of-a-look question with the memory itself as the currency.

- Nonlinear writes (resource depletion à la Tsodyks–Markram) would make G3 more than a
  matched filter — writes would interact, and spreading vs stacking pulses would matter.
- The chessboard pinger (homeostatic coupler) learning probe trains online across a
  whole graph of residue routes, rather than one route at a time.
