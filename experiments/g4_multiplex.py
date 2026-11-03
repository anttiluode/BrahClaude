"""G4 — two messages down one axon; the routes' residue does the separating.

Event rate a(t) and burst probability b(t) are independent slow signals. One spike
train carries both. A depressing target sees ~events (a); a facilitating target sees
events weighted by burstiness; their ratio recovers b. (Naud & Sprekeler 2018 shape.)
"""
import numpy as np
from scipy.signal import lfilter

from common import save
from brah.edges import run

T, TAU_SIG, TAU_LP, REFR = 200_000, 1000, 500, 8
GD = np.array([-1.6, -0.8, 0, 0, 0, 0, 0, 0.0])
GF = np.array([2.0, 1.0, 0, 0, 0, 0, 0, 0.0])


def ou(rng, n, tau):
    a = np.exp(-1 / tau)
    return lfilter([np.sqrt(1 - a * a)], [1, -a], rng.normal(size=n))


def lowpass(x, tau):
    a = np.exp(-1 / tau)
    return lfilter([1 - a], [1, -a], x, axis=0)


def sig(z):
    return 1 / (1 + np.exp(-z))


def one_seed(seed):
    rng = np.random.default_rng(seed)
    a = 0.005 + 0.025 * sig(1.5 * ou(rng, T, TAU_SIG))
    b = 0.1 + 0.8 * sig(1.5 * ou(rng, T, TAU_SIG))
    x = np.zeros(T)
    onsets = np.zeros(T)  # ground-truth event onsets (diagnostic ceiling only)
    t, last = 0, -REFR - 1
    while t < T - 6:
        if t - last > REFR and rng.random() < a[t]:
            n = 3 if rng.random() < b[t] else 1
            x[t + 2 * np.arange(n)] = 1.0
            onsets[t] = 1.0
            last = t + 2 * (n - 1)
            t = last + 1
        else:
            t += 1
    G = np.stack([GD, GF, np.zeros(8), np.zeros(8)])
    y, _ = run(np.tile(x[:, None], (1, 4)), G)
    lp = lowpass(y, TAU_LP)
    burn = 5 * TAU_LP
    a_ref, b_ref = lowpass(a, TAU_LP)[burn:], lowpass(b, TAU_LP)[burn:]
    D, F, F0, D0 = lp[burn:, 0], lp[burn:, 1], lp[burn:, 2], lp[burn:, 3]
    # after burn-in every low-passed readout is strictly positive, so plain ratios
    # (first run used F/(D+1e-6), which made the flat control spuriously non-constant)
    assert D.min() > 0 and D0.min() > 0
    a_hat, b_hat = D, F / D
    b_flat = F0 / D0
    c = lambda u, v: float(np.corrcoef(u, v)[0, 1])
    flat_const = bool(np.ptp(b_flat) < 1e-9)
    ideal = lowpass(onsets, TAU_LP)[burn:]
    return {
        "ceiling_ideal_event_counter_a": c(ideal, a_ref),
        "corr_ahat_ideal_counter": c(a_hat, ideal),
        "corr_ahat_a": c(a_hat, a_ref), "corr_bhat_b": c(b_hat, b_ref),
        "corr_ahat_b": c(a_hat, b_ref), "corr_bhat_a": c(b_hat, a_ref),
        "flat_bhat_constant": flat_const,
        "corr_flat_bhat_b": 0.0 if flat_const else c(b_flat, b_ref),
        # what a residue-free reader sees: raw rate mixes a and b
        "corr_rawrate_a": c(F0, a_ref), "corr_rawrate_b": c(F0, b_ref),
    }


def main():
    per = [one_seed(s) for s in range(5)]
    mean = {k: float(np.mean([p[k] for p in per])) for k in per[0] if k != "flat_bhat_constant"}
    mean["flat_bhat_constant_all"] = all(p["flat_bhat_constant"] for p in per)
    passed = (mean["corr_ahat_a"] >= 0.8 and mean["corr_bhat_b"] >= 0.7
              and abs(mean["corr_ahat_b"]) <= 0.25 and abs(mean["corr_bhat_a"]) <= 0.25
              and (mean["flat_bhat_constant_all"] or abs(mean["corr_flat_bhat_b"]) <= 0.1))
    res = {"per_seed": per, "mean": mean, "pass": passed}
    print(res["mean"], passed)
    save("g4", res)


if __name__ == "__main__":
    main()
