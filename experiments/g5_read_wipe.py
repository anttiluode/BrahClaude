"""G5 — is a ping read destructive?  (pre-registered in PREREGISTRATION.md, commit bc677bb)

LIN: the temporal-Sihti AUG edge (linear history before the clip).
TM : Tsodyks-Markram facilitating/depressing route (multiplicative, saturating).
Arms: N neutral reads, 0 one time-matched read, S selective refresh (oracle).
"""
import itertools

import numpy as np

from common import save
from brah import tm
from brah.edges import TYPES, run as lin_run

E, NREAD, T_FIRST = 8, 12, 155
PERIODS = (2, 10, 40)
BURST = (0, 2, 4)
NTRIAL = 20000

# the G2a item draw, exactly (rng seed 2, first use)
_rng_items = np.random.default_rng(2)
pairs = list(itertools.combinations(range(E), 2))
ITEMS = [pairs[i] for i in _rng_items.choice(len(pairs), 8, replace=False)]
rng = np.random.default_rng(5)


def route(model, x):
    if model == "LIN":
        y, _ = lin_run(x, np.tile(TYPES["AUG"], (E, 1)))
        return y
    y, _ = tm.run(x)
    return y


def trains(arm, P):
    """Pulse trains (item, T, E) and the read times for one arm."""
    reads = [T_FIRST + P * i for i in range(NREAD)]
    T = reads[-1] + 1
    out = []
    for it in ITEMS:
        x = np.zeros((T, E))
        for r in it:
            x[list(BURST), r] = 1.0
        if arm == "N":
            x[reads] = 1.0
        elif arm == "0":
            x[reads[-1]] = 1.0
        elif arm == "S":
            for t in reads[:-1]:
                x[t, list(it)] = 1.0
            x[reads[-1]] = 1.0
        out.append(x)
    return np.array(out), reads


def responses(model, arm, P):
    X, reads = trains(arm, P)
    return np.array([route(model, x)[reads] for x in X]), reads  # (item, read, E)


def acc(templates, sigma):
    lab = rng.integers(0, len(templates), NTRIAL)
    obs = templates[lab] + rng.normal(0, sigma, (NTRIAL, E))
    d2 = ((obs[:, None, :] - templates[None]) ** 2).sum(-1)
    return float((d2.argmin(1) == lab).mean())


def gap(Y):
    """mean written-minus-unwritten response, per read (Y: item, read, E)."""
    g = []
    for k in range(Y.shape[1]):
        w = np.mean([Y[i, k, list(it)].mean() for i, it in enumerate(ITEMS)])
        u = np.mean([np.delete(Y[i, k], list(it)).mean() for i, it in enumerate(ITEMS)])
        g.append(w - u)
    return np.array(g)


def diffs(Y):
    """written-minus-unwritten per item (item, read)."""
    return np.array([[Y[i, k, list(it)].mean() - np.delete(Y[i, k], list(it)).mean()
                      for k in range(Y.shape[1])] for i, it in enumerate(ITEMS)])


res = {"items": ITEMS}
# G5-0a: LIN common-mode identity (each read of arm N vs a lone read at the same time)
err = 0.0
for P in PERIODS:
    YN, reads = responses("LIN", "N", P)
    dN = diffs(YN)
    for k, t in enumerate(reads):
        X, _ = trains("0", P)
        lone = []
        for x in X:
            x2 = x[: t + 1].copy(); x2[T_FIRST:] = 0.0; x2[t] = 1.0
            lone.append(route("LIN", x2)[t])
        lone = np.array(lone)[:, None, :]
        err = max(err, float(np.abs(dN[:, k] - diffs(lone)[:, 0]).max()))
res["G5-0a_max_err"] = err
res["G5-0a"] = err < 1e-9
y_fresh, _ = tm.run(np.ones((1, E)))
res["G5-0b_max_err"] = float(np.abs(y_fresh - 1.0).max())
res["G5-0b"] = bool(np.all(y_fresh == 1.0))   # as pre-registered: exact equality

for model in ("LIN", "TM"):
    YN2, _ = responses(model, "N", 2)
    sigma = 0.25 * gap(YN2)[0]
    res[model] = {"sigma": sigma}
    for P in PERIODS:
        YN, _ = responses(model, "N", P)
        Y0, _ = responses(model, "0", P)
        YS, _ = responses(model, "S", P)
        accN = [acc(YN[:, k], sigma) for k in range(NREAD)]
        acc0 = acc(Y0[:, -1], sigma)   # arm 0's only pulse is at t_11 (bug fix: was index 0)
        accS = acc(YS[:, -1], sigma)
        res[model][f"P{P}"] = {
            "acc_N_by_read": accN, "acc_0": acc0, "acc_S": accS,
            "read_cost": acc0 - accN[-1],
            "gap_N_first_last": [gap(YN)[0], gap(YN)[-1]],
            "gap_0": gap(Y0)[-1], "gap_S": gap(YS)[-1],
        }
        r = res[model][f"P{P}"]
        print(f"{model} P={P:2d}  N reads {np.round(accN[:1] + accN[3:4] + accN[-1:], 3)} (1st,4th,12th)"
              f"  0 {acc0:.3f}  S {accS:.3f}  cost {r['read_cost']:+.3f}"
              f"  gap N {r['gap_N_first_last'][0]:.3f}->{r['gap_N_first_last'][1]:.3f}"
              f" 0 {r['gap_0']:.3f} S {r['gap_S']:.3f}")

L, M = res["LIN"], res["TM"]
res["G5-1"] = all(L[f"P{P}"]["read_cost"] <= 0.02 for P in PERIODS)
res["G5-2"] = M["P2"]["read_cost"] >= 0.30
res["G5-3"] = M["P40"]["read_cost"] < M["P2"]["read_cost"]
res["G5-4"] = all(M[f"P{P}"]["acc_S"] >= M[f"P{P}"]["acc_0"] for P in PERIODS)
print({k: res[k] for k in ("G5-0a", "G5-0a_max_err", "G5-0b", "G5-0b_max_err", "G5-1", "G5-2", "G5-3", "G5-4")})
save("g5", res)
