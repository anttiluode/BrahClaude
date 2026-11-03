# Pre-registration — temporal-Sihti edges (BrahClaude v0)

Written and committed **before any gate was run**. The only thing computed before
this file was a scale check of the four edge kernels kappa(s) (a model property, not
an outcome): FAC peaks at s=1, DEP is most negative at s=1, BAND peaks at s*=13,
AUG peaks at s=52.

Model: `brah/edges.py`. K=8 octave bands (tau = 2..256 steps). Pulse out
`y = x * max(0, 1 + g.h)`, h = Sihti residue stack of the edge's own pulse history.
Edge types FAC / DEP / BAND / AUG as in `brah/edges.py::TYPES`.

The question (from the Nov 3 chat): if a pulse carries the route's Sihti residue of
earlier pulses, can a neuron (a) identify routes by probing, (b) read silent memory
left by someone else's pulses, (c) learn to write residue for later, and (d) send
two messages down one axon that the routes separate?

## G0 — identities (must hold to float precision; otherwise the model is wrong)
- G0a telescoping: sum_k h_k == m_{K-1} for random trains, max err < 1e-9.
- G0b unclipped edge == explicit 2nd-order Volterra with kappa(s), max err < 1e-9.
- G0c the first pulse on a fresh edge returns exactly 1 for every type (carries no
  information about the edge: reading requires a prior write).

## G1 — identify the hidden edge type by probing (paired-pulse generalised)
4 equiprobable types, 6 pulses inside 128 steps, Gaussian release noise sigma=0.3 on
each response, Bayes decoder with known templates, accuracy by Monte Carlo (fresh
noise, 20 000 trials). Fixed trains: paired (ISI 3, three pairs), regular (ISI 21),
burst (ISI 1), random placement (mean of 200 draws). Optimised train: cross-entropy
search over pulse positions, scored on a separate noise sample.
- PASS if optimised accuracy >= 0.90 AND >= (best fixed or random-mean) + 0.05.

## G2 — silent memory read by a neutral ping (Wolff-style)
- G2a: 8 routes, all AUG. 8 items, each = a 3-spike burst (ISI 2) on 2 of the 8
  routes. 150 silent steps, then one ping spike on all routes, noise sigma=0.1.
  PASS if: ping decoding accuracy >= 0.90; total output during the delay is exactly
  0 (memory is activity-silent); with residues reset to the across-route mean before
  the ping, accuracy <= 0.175 (chance 0.125 + 0.05).
- G2b amount vs age: one route, 4 terminals. Burst size n in {1,2,3,4} (ISI 2) and
  age D in {20,40,80,160} -> 16 classes; one ping, sigma=0.05.
  Heterogeneous terminals: g=0.3 on band pairs (3,4),(4,5),(5,6),(6,7).
  Homogeneous: all four = mean of those vectors.
  PASS if hetero accuracy >= 0.80 AND homo accuracy <= 0.50
  (one band projection confounds "many pulses long ago" with "few pulses recently").

## G3 — learned write-for-later
Two routes A (target) and B to a readout. Pinger places exactly 6 pulses in
[0, 200) on A and/or B; a third-party spike arrives on both routes at
t_s ~ U[200, 200+W). Reward = E[y_A(t_s) - y_B(t_s)] (computed exactly over t_s).
Policy learned by cross-entropy method with no model access (reward only).
- G3a FAC (W=32): >= 80% of learned pulses on A. DEP (W=32): >= 80% on B.
- G3b BAND: learned mean lead time (200 - pulse time) at W=4 within +-25% of
  s*=13, and lead time at W=128 is smaller than at W=4.
- G3c learned reward >= 0.95 x best of (greedy top-slot policy) for every case run.

## G4 — two messages down one axon, separated by the routes (burst multiplexing)
Sender: events at slowly varying rate a(t) (0.005-0.03/step, refractory 8 steps),
each event a 3-spike burst (ISI 2) with slowly varying probability b(t) (0.1-0.9);
a, b independent OU-driven, timescale 1000 steps, T=200 000, 5 seeds.
Targets on the same axon: Dx g=[-1.6,-0.8,0..] and Fx g=[2.0,1.0,0..]; outputs
low-passed (tau=500). Decoded a_hat = D readout, b_hat = F/D readout.
- PASS if mean corr(a_hat, a) >= 0.8, corr(b_hat, b) >= 0.7,
  |corr(a_hat, b)| <= 0.25, |corr(b_hat, a)| <= 0.25,
  and with flat targets (g=0) b_hat carries no information (|corr| <= 0.1 or constant).

Every gate is reported as pre-registered, pass or fail.

---

# G5 — is a ping read destructive? (pre-registered 3 Nov 2026, before any G5 run)

**Why.** In `brah/edges.py` a neutral ping writes the same pulse on every route, so
before the clip it shifts every route's response equally and the differences that
hold the memory are untouched: reading is free. Real short-term plasticity is
multiplicative and saturating (Tsodyks–Markram), so reading may erase what it reads.
This also ties to `pinger/` Gate 4B, where probing a plastic medium changed it.

**Disclosed pilot.** An informal run in the chat used different settings: U = 0.05,
τ_F = 300, τ_D = 20 steps, output normalised by U, σ = 0.1 for both models, and arm N
only. Over 12 reads at period 2 it gave LIN accuracy 0.996 → 0.989 and TM 1.00 → 0.29.
The predictions below are informed by it. Changed since the pilot: U, τ_D, the
normalisation, the σ rule, and two new arms (0 and S).

**Routes.** LIN = 8 AUG edges from `brah/edges.py`, exactly as G2a. TM = `brah/tm.py`
with U = 0.2, τ_F = 300, τ_D = 40 steps (Mongillo et al. 2008 proportions at 5 ms per
step). Its output is normalised so a fresh synapse answers 1. The only model property
checked before this file: a fresh TM route answers exactly 1, and a 3-spike burst at
ISI 2 releases 1, 0.89, 0.60.

**Task.** The G2a items (the same seed-2 draw: 8 items, each a 3-spike burst at
t = 0, 2, 4 on 2 of the 8 routes). Reads happen at t_i = 155 + P·i, i = 0..11,
P ∈ {2, 10, 40}.

**Arms.**
- **N (neutral reads):** every read is a pulse on all 8 routes, and each read is decoded on its own.
- **0 (time-matched, no prior reads):** one neutral read at t_11 only.
- **S (selective refresh, oracle):** reads 0–10 pulse only the item's written routes (the reader knows the item, i.e. it has an efference copy). Read 11 is neutral and is the one decoded.

**Decoding.** Bayes nearest template, with templates per arm and per read index (the
reader is given the dynamics, and that gift is stated). Isotropic Gaussian noise, with
σ per route model = 0.25 × the mean written-minus-unwritten response at read 0 of arm
N. 20 000 Monte Carlo trials, rng seed 5.
**Read cost** := acc(arm 0, t_11) − acc(arm N, read 11).

**Criteria.**
- G5-0a identity: for LIN, written-minus-unwritten differences at every read of arm N equal those of a lone read at the same time. Max abs error < 1e-9 (a common-mode write).
- G5-0b identity: a fresh TM route's first pulse returns exactly 1.
- G5-1 LIN reads are free: read cost ≤ 0.02 at every P.
- G5-2 TM dense reads are destructive: read cost ≥ 0.30 at P = 2.
- G5-3 TM read cost shrinks with spacing: cost(P = 40) < cost(P = 2).
- G5-4 TM selective refresh preserves the memory: acc(S, read 11) ≥ acc(0, t_11) at every P.

Every criterion is reported as pre-registered, pass or fail.

---

# G6 — addresses through one soma, when every look costs memory (pre-registered 3 Nov 2026, before any G6 run)

**Why.** Sol (BrahSol, 3 Nov) pointed out that G2a's 99.6% needs eight separately
recorded route outputs. Summed into one soma, which is what the original coupler did
with its four taps, every item gives the same total, because every item has exactly two
written routes. Sol also showed that three route subsets separate G2a's eight items
when there is no noise. G6 asks the general version: through **one summed readout**, can a
prober find which routes were written by choosing **which routes to ping** (an address
as a sequence of interventions)? And what does that questioning cost the memory, on a
linear route (`brah/edges.py`, AUG) versus a Tsodyks–Markram route (`brah/tm.py`, as in G5)?

**World.** 8 routes. The hidden state is one of all **28** route pairs (uniform prior;
not the 8 G2a items, so the design cannot be tuned to them). It is written by a 3-spike
burst at t = 0, 2, 4 on both routes of the pair. Queries happen at t_q = 155 + P·q,
q = 0..11, with P ∈ {10, 40}.

**Query.** The prober picks a nonempty route subset S and pulses those routes. It
observes one number, z = Σ_{e∈S} y_e + ε, with ε ~ N(0, σ²) and the same σ whatever
the size of S (one soma, one noise source). σ per model = 0.25 × the single-route
written-minus-unwritten response at t = 155 on a fresh written route (the G5 rule).

**Belief.** Exact Bayes over the 28 hypotheses. The prober is given the route dynamics
and its own query history (an efference copy); this gift is stated.

**Policies.**
- **B** broadcast: S = all 8.
- **R** random: S is a uniform random subset of size 4.
- **G** greedy: S among all 255 nonempty subsets, maximising the mutual information between the hypothesis and z, estimated by Monte Carlo with 400 samples per subset.
- **O** reference: 8 separate channels with broadcast pings (not one soma).

1000 episodes per model × policy × P, rng seed 6.

**Memory left.** After the 12th query, wait P steps, then make one 8-channel broadcast
diagnostic read with the same σ. Its Bayes accuracy over the 28 pairs is computed with
the true per-hypothesis templates. The unqueried baseline is the same diagnostic read
at the same time with no prior queries.

**Criteria.**
- **G6-0 (exact):** under B the posterior stays equal to the prior after every query, for both models. Max abs deviation < 1e-9.
- **G6-1:** LIN, P = 10: G reaches MAP accuracy ≥ 0.90 by query 12, and G ≥ R + 0.10 at query 6.
- **G6-2:** TM, P = 10: G's accuracy at query 12 is lower than LIN's G at query 12 (destructive reads limit what one soma can learn).
- **G6-3:** P = 10: the questioning costs the TM memory, with diagnostic accuracy after G's 12 queries ≥ 0.20 below the unqueried baseline. For LIN it costs nothing, with |difference| ≤ 0.02.
- **Descriptive only (no criterion):** subset sizes G chooses, and how often it re-pings a route it pinged in the previous query, LIN vs TM.

## Post-hoc diagnostics (requested by Sol after G2b/G5 were run; not pre-registered, reported as such)
- **D1 (G2b matched raw-trace baseline):** 4 terminals that each read one raw trace m_k (k = 3, 4, 5, 6) instead of band pairs, scaled to the same mean response size as the hetero terminals, with the same σ and decoder.
- **D2 (G5 hidden state):** for the TM route, the written-minus-unwritten distance in hidden state (u, R) at t_11, arm N vs arm 0. This tests whether erasure is in the state itself, not only in the readout.
