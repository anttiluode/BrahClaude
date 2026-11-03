"""G6 — addresses through one soma, when every look costs memory.
Pre-registered in PREREGISTRATION.md (commit 83a50a4), before this file was run.

8 routes, hidden = which of the 28 route pairs was written (3-spike burst).
Prober pings a route subset S and sees ONE summed number z = sum_S y + noise.
Exact Bayes over 28 hypotheses; the prober is given the dynamics and its own history.
"""
import itertools
import os
import sys

import numpy as np

from common import save
from brah import tm
from brah.edges import TYPES, lams, run as lin_run

E, NQ, T0 = 8, 12, 155
PAIRS = list(itertools.combinations(range(E), 2))
H = len(PAIRS)
SUBSETS = np.array([[(s >> e) & 1 for e in range(E)] for s in range(1, 2 ** E)], float)  # 255 x 8
EPIS = int(os.environ.get("G6_EPISODES", 1000))
MC = 400
G_AUG = TYPES["AUG"]
LAM = lams(len(G_AUG))
U, DF, DD = tm.U_DEF, np.exp(-1 / tm.TAU_F_DEF), np.exp(-1 / tm.TAU_D_DEF)
Y0 = 2 * U - U * U


# ---------- route state for all 28 hypotheses at once: arrays (H, E[, K]) ----------
class Lin:
    name = "LIN"
    def init(self):
        x = np.zeros((T0, E * H))
        for h, (i, j) in enumerate(PAIRS):
            for r in (i, j):
                x[[0, 2, 4], h * E + r] = 1.0
        _, m = lin_run(x, np.tile(G_AUG, (E * H, 1)))
        return m.reshape(H, E, -1)
    def respond(self, s):            # answer of every route if pinged now
        h = s.copy(); h[..., 1:] = s[..., 1:] - s[..., :-1]
        return np.maximum(0.0, 1.0 + (h * G_AUG).sum(-1))
    def commit(self, s, S):          # one step with pulses on S
        return LAM * s + S[None, :, None]
    def wait(self, s, n):
        return s * LAM ** n


class TM:
    name = "TM"
    def init(self):
        x = np.zeros((T0, E * H))
        for h, (i, j) in enumerate(PAIRS):
            for r in (i, j):
                x[[0, 2, 4], h * E + r] = 1.0
        _, (u, R) = tm.run(x)
        return np.stack([u.reshape(H, E), R.reshape(H, E)], -1)
    def respond(self, s):
        u, R = s[..., 0], s[..., 1]
        return (u + U * (1 - u)) * R / Y0
    def commit(self, s, S):
        u, R = s[..., 0], s[..., 1]
        u_sp = u + U * (1 - u) * S
        R = R - u_sp * R * S
        u = U + (u_sp - U) * DF
        R = 1 - (1 - R) * DD
        return np.stack([u, R], -1)
    def wait(self, s, n):
        u, R = s[..., 0], s[..., 1]
        return np.stack([U + (u - U) * DF ** (n), 1 - (1 - R) * DD ** (n)], -1)


def validate(model):
    """state stepping == the reference route code, on one random history."""
    rng = np.random.default_rng(0)
    s = model.init()
    x = np.zeros((T0 + 60, E * H))
    for h, (i, j) in enumerate(PAIRS):
        for r in (i, j):
            x[[0, 2, 4], h * E + r] = 1.0
    ys = []
    for t in range(T0, T0 + 60):
        S = (rng.random(E) < 0.3).astype(float)
        x[t] = np.tile(S, H)
        ys.append(model.respond(s) * S[None])
        s = model.commit(s, S)
    if model.name == "LIN":
        yref, _ = lin_run(x, np.tile(G_AUG, (E * H, 1)))
    else:
        yref, _ = tm.run(x)
    return float(np.abs(np.array(ys).reshape(60, -1) - yref[T0:]).max())


def mi_greedy(mu, w, sigma, rng):
    """argmax_S I(h; z) for z | h ~ N(mu[S,h], sigma^2), h ~ w.  mu: (255, H)."""
    hs = rng.choice(H, MC, p=w)
    z = mu[:, hs] + sigma * rng.standard_normal((1, MC))              # (255, MC)
    ll = -0.5 * ((z[:, :, None] - mu[:, None, :]) / sigma) ** 2         # (255, MC, H)
    lw = np.log(w + 1e-300)[None, None, :]
    a = ll + lw
    mx = a.max(-1, keepdims=True)
    logpz = mx[..., 0] + np.log(np.exp(a - mx).sum(-1))                 # log p(z) (up to const)
    logpz_h = ll[np.arange(255)[:, None], np.arange(MC)[None, :], hs[None, :]]
    return int(np.argmax((logpz_h - logpz).mean(1)))


def episode(model, policy, P, sigma, rng, s_init):
    truth = int(rng.integers(H))
    s = s_init.copy()
    logp = np.full(H, -np.log(H))
    acc, maxdev, sizes, reping = [], 0.0, [], []
    prev = None
    for q in range(NQ):
        r = model.respond(s)                                         # (H, E)
        if policy == "O":
            obs = r[truth] + sigma * rng.standard_normal(E)
            logp += -0.5 * (((obs[None] - r) / sigma) ** 2).sum(1)
            S = np.ones(E)
        else:
            if policy == "B":
                S = np.ones(E)
            elif policy == "R":
                S = np.zeros(E); S[rng.choice(E, 4, replace=False)] = 1.0
            else:
                w = np.exp(logp - logp.max()); w /= w.sum()
                S = SUBSETS[mi_greedy(SUBSETS @ r.T, w, sigma, rng)]
            mu = r @ S
            z = mu[truth] + sigma * rng.standard_normal()
            logp += -0.5 * ((z - mu) / sigma) ** 2
        logp -= np.logaddexp.reduce(logp)
        if policy == "B":
            maxdev = max(maxdev, float(np.abs(np.exp(logp) - 1 / H).max()))
        acc.append(int(np.argmax(logp) == truth))
        sizes.append(S.sum())
        if prev is not None:
            reping.append(float((S * prev).sum() / S.sum()))
        prev = S
        s = model.commit(s, S)
        s = model.wait(s, P - 1)
    # memory left: one 8-channel broadcast read P steps after the last query
    templ = model.respond(s)                                          # (H, E), at t_11 + P
    obs = templ[truth][None] + sigma * rng.standard_normal((20, E))
    d2 = ((obs[:, None, :] - templ[None]) ** 2).sum(-1)
    diag = float((d2.argmin(1) == truth).mean())
    return np.array(acc), maxdev, diag, float(np.mean(sizes)), float(np.mean(reping))


def baseline_diag(model, P, sigma, rng, s_init, n=20000):
    s = model.wait(s_init, P * NQ)                                    # t_11 + P with no queries
    templ = model.respond(s)
    truth = rng.integers(0, H, n)
    obs = templ[truth] + sigma * rng.standard_normal((n, E))
    d2 = ((obs[:, None, :] - templ[None]) ** 2).sum(-1)
    return float((d2.argmin(1) == truth).mean())


if __name__ == "__main__":
    rng = np.random.default_rng(6)
    res = {}
    for model in (Lin(), TM()):
        s0 = model.init()
        r0 = model.respond(s0)
        h01 = PAIRS.index((0, 1))
        sigma = 0.25 * float(r0[h01, 0] - r0[h01, 7])   # written route 0 minus unwritten route 7
        res[model.name] = {"sigma": sigma, "stepping_vs_reference_max_err": validate(model)}
        for P in (10, 40):
            res[model.name][f"P{P}"] = {"diag_unqueried": baseline_diag(model, P, sigma, rng, s0)}
            for pol in ("B", "R", "G", "O"):
                A, dev, D, sz, rp = [], 0.0, [], [], []
                for _ in range(EPIS):
                    a, m, d, s_, r_ = episode(model, pol, P, sigma, rng, s0)
                    A.append(a); dev = max(dev, m); D.append(d); sz.append(s_); rp.append(r_)
                A = np.array(A).mean(0)
                out = {"acc_by_query": A.round(4).tolist(), "diag_after": float(np.mean(D)),
                       "mean_subset_size": float(np.mean(sz)), "reping_frac": float(np.mean(rp))}
                if pol == "B":
                    out["max_posterior_dev_from_prior"] = dev
                res[model.name][f"P{P}"][pol] = out
                print(f"{model.name} P={P:2d} {pol}: acc q1 {A[0]:.3f} q6 {A[5]:.3f} q12 {A[-1]:.3f}"
                      f" | memory left {np.mean(D):.3f} (unqueried {res[model.name][f'P{P}']['diag_unqueried']:.3f})"
                      f" | |S| {np.mean(sz):.2f} re-ping {np.mean(rp):.2f}"
                      + (f" | maxdev {dev:.1e}" if pol == "B" else ""), flush=True)
    L, M = res["LIN"]["P10"], res["TM"]["P10"]
    res["G6-0"] = max(res[m][f"P{P}"]["B"]["max_posterior_dev_from_prior"]
                      for m in ("LIN", "TM") for P in (10, 40)) < 1e-9
    res["G6-1"] = L["G"]["acc_by_query"][-1] >= 0.90 and L["G"]["acc_by_query"][5] >= L["R"]["acc_by_query"][5] + 0.10
    res["G6-2"] = M["G"]["acc_by_query"][-1] < L["G"]["acc_by_query"][-1]
    res["G6-3"] = (M["diag_unqueried"] - M["G"]["diag_after"] >= 0.20
                   and abs(L["diag_unqueried"] - L["G"]["diag_after"]) <= 0.02)
    print({k: res[k] for k in ("G6-0", "G6-1", "G6-2", "G6-3")},
          "stepping err", res["LIN"]["stepping_vs_reference_max_err"], res["TM"]["stepping_vs_reference_max_err"])
    save("g6", res)
