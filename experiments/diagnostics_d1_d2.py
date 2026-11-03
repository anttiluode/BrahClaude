"""Post-hoc diagnostics requested by Sol after G2b and G5 were run.
Declared in PREREGISTRATION.md (commit 83a50a4) as NOT pre-registered.

D1  G2b matched raw-trace baseline: does the Sihti band basis beat 4 terminals that
    each read one raw exponential trace m_k (k = 3..6), scaled to the same mean size?
D2  G5 hidden state: on the TM route, is the memory erased in (u, R) itself, or only
    in the readout?  Written-minus-unwritten state distance at t_11, arm N vs arm 0.
"""
import numpy as np

from common import bayes_accuracy, save
from brah import tm
from brah.edges import bands_from_traces, lams, run

rng = np.random.default_rng(7)
out = {}

# ---------------- D1 ----------------
SIG_B = 0.05
classes = [(n, d) for n in (1, 2, 3, 4) for d in (20, 40, 80, 160)]
hetero = np.zeros((4, 8))
for i, (a, b) in enumerate([(3, 4), (4, 5), (5, 6), (6, 7)]):
    hetero[i, [a, b]] = 0.3
homo = np.tile(hetero.mean(0), (4, 1))


def traces_before_ping(n, d):
    x = np.zeros((2 * (n - 1) + d, 1))
    x[2 * np.arange(n)] = 1.0
    _, m = run(x, np.zeros(8))
    return m[0]                                   # raw traces m_k just before the ping


M = np.array([traces_before_ping(n, d) for n, d in classes])        # (16, 8)
Hb = bands_from_traces(M)
resp_hetero = 1 + Hb @ hetero.T                                     # (16, 4) same as G2b
resp_homo = 1 + Hb @ homo.T
raw = np.zeros((4, 8))
for i, k in enumerate((3, 4, 5, 6)):
    raw[i, k] = 1.0
proj = M @ raw.T                                                   # one raw trace per terminal
scale = np.abs(resp_hetero - 1).mean() / np.abs(proj).mean()
resp_raw = 1 + scale * proj
out["D1"] = {
    "hetero_bands": bayes_accuracy(resp_hetero, SIG_B, 20000, rng),
    "homogeneous": bayes_accuracy(resp_homo, SIG_B, 20000, rng),
    "raw_traces_matched": bayes_accuracy(resp_raw, SIG_B, 20000, rng),
    "raw_scale": float(scale),
}
print("D1", out["D1"])

# ---------------- D2 ----------------
import itertools  # noqa: E402
E, T_FIRST, NREAD = 8, 155, 12
_r = np.random.default_rng(2)
pairs = list(itertools.combinations(range(E), 2))
ITEMS = [pairs[i] for i in _r.choice(len(pairs), 8, replace=False)]
d2 = {}
for P in (2, 10, 40):
    reads = [T_FIRST + P * i for i in range(NREAD)]
    for arm in ("N", "0"):
        dist = []
        for it in ITEMS:
            x = np.zeros((reads[-1], E))            # stop just BEFORE the 12th read
            for r in it:
                x[[0, 2, 4], r] = 1.0
            if arm == "N":
                x[reads[:-1]] = 1.0
            _, (u, R) = tm.run(x)
            w = list(it); nw = [e for e in range(E) if e not in it]
            dist.append([u[w].mean() - u[nw].mean(), R[w].mean() - R[nw].mean()])
        d2[f"P{P}/{arm}"] = np.mean(dist, 0).round(5).tolist()
    print(f"D2 P={P}: written-minus-unwritten (u, R) before read 12: "
          f"after 11 neutral reads {d2[f'P{P}/N']}, unread {d2[f'P{P}/0']}")
out["D2"] = d2
save("diagnostics_d1_d2", out)
