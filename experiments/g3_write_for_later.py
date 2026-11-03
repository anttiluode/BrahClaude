"""G3 — does a reward-only learner discover writing residue for a later pulse?

Routes A (target) and B. The pinger places 6 pulses before t0=200. A third-party
spike then arrives on both routes at t_s ~ U[t0, t0+W). Reward = E[y_A - y_B].
The learner (cross-entropy method) sees only rewards, never the edge model.
"""
import numpy as np

from common import save
from brah.edges import TYPES, kernel

T0, P, SMAX = 200, 6, 600


def reward(slots, kap, W):
    """slots: indices in [0, 2*T0); <T0 -> route A at that time, else route B."""
    slots = np.asarray(slots)
    ts = T0 + np.arange(W)
    out = 0.0
    ys = []
    for route in (0, 1):
        p = slots[(slots >= route * T0) & (slots < (route + 1) * T0)] - route * T0
        if len(p):
            lag = ts[:, None] - p[None, :]
            ys.append(np.maximum(0.0, 1.0 + kap[lag - 1].sum(1)))
        else:
            ys.append(np.ones(W))
    return float((ys[0] - ys[1]).mean())


def cem(kap, W, rng, iters=150, pop=200, elite=20):
    logits = np.zeros(2 * T0)
    best, best_s = -np.inf, None
    for _ in range(iters):
        cands, sc = [], []
        for _ in range(pop):
            s = np.argsort(-(logits + rng.gumbel(size=2 * T0)))[:P]
            cands.append(s); sc.append(reward(s, kap, W))
        order = np.argsort(sc)[::-1][:elite]
        if sc[order[0]] > best:
            best, best_s = sc[order[0]], np.sort(cands[order[0]])
        freq = np.zeros(2 * T0)
        for i in order:
            freq[cands[i]] += 1
        p = 0.6 * np.exp(logits - logits.max()); p /= p.sum()
        p = p + 0.4 * freq / freq.sum()
        logits = np.log(p + 1e-12)
    return best_s, best


def greedy(kap, W):
    chosen = []
    for _ in range(P):
        free = [s for s in range(2 * T0) if s not in chosen]
        r = [reward(chosen + [s], kap, W) for s in free]
        chosen.append(free[int(np.argmax(r))])
    return np.sort(chosen), reward(chosen, kap, W)


def describe(slots):
    slots = np.asarray(slots)
    onA = slots[slots < T0]
    onB = slots[slots >= T0] - T0
    allt = np.concatenate([onA, onB])
    return {"on_A": onA.tolist(), "on_B": onB.tolist(),
            "frac_A": len(onA) / P, "mean_lead": float((T0 - allt).mean())}


def main():
    rng = np.random.default_rng(3)
    cases = [("FAC", 32), ("DEP", 32), ("BAND", 4), ("BAND", 32), ("BAND", 128), ("AUG", 32)]
    out = {}
    for name, W in cases:
        kap = kernel(TYPES[name], SMAX)
        s, r = cem(kap, W, rng)
        gs, gr = greedy(kap, W)
        out[f"{name}_W{W}"] = {"learned": describe(s), "learned_reward": r,
                               "greedy": describe(gs), "greedy_reward": gr,
                               "kappa_peak_lag": int(np.argmax(kap) + 1)}
        print(name, W, out[f"{name}_W{W}"])
    lead4 = out["BAND_W4"]["learned"]["mean_lead"]
    lead128 = out["BAND_W128"]["learned"]["mean_lead"]
    passes = {
        "G3a": out["FAC_W32"]["learned"]["frac_A"] >= 0.8 and out["DEP_W32"]["learned"]["frac_A"] <= 0.2,
        "G3b": abs(lead4 - 13) <= 0.25 * 13 and lead128 < lead4,
        "G3c": all(v["learned_reward"] >= 0.95 * v["greedy_reward"] for v in out.values()),
    }
    out["pass"] = passes
    print(passes)
    save("g3", out)


if __name__ == "__main__":
    main()
