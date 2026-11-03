"""
The pinger: the thing that sits where the Homeostatic Coupler sat.

It sends one ping per tick (one of K levels in [0,1]), gets four taps back,
and keeps a MEMORY TABLE: for every ping it has tried, an exponentially
forgetting estimate of what came back, how wrong its last guess was, and when
it last asked.  That table is "what the pinger knows about the medium".

Policies (what decides the next ping):
  coupler   the original edge-of-chaos coupler, untouched; it never reads the
            table - it only regulates the variance of the summed return.
            The table is filled passively from whatever it happens to ping.
  random    uniform over pings.
  sweep     0,1,2,...,K-1,0,1,...
  curious   ask where the expected error is largest:
            U_k = (recent squared error at k + eps) * (ticks since k was asked)
  dual      homeostasis AND curiosity, using its own table to predict:
            score_k = ((var_if_k - target)/target)^2 - lam * U_k / max U
"""
from collections import deque
import numpy as np
from .ecg_loop import Coupler

K = 32
LEVELS = np.linspace(0.0, 1.0, K)
TARGET_VAR, WINDOW = 0.1, 50


class Table:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.y = np.zeros((K, 4)); self.n = np.zeros(K, int)
        self.err = np.zeros(K); self.last = np.full(K, -10**9)
    def predict(self):
        out = self.y.copy()
        seen = self.n > 0
        fallback = self.y[seen].mean(0) if seen.any() else np.zeros(4)
        out[~seen] = fallback
        return out
    def update(self, k, y, t):
        guess = self.predict()[k]
        e2 = float(np.sum((y - guess) ** 2))
        if self.n[k] == 0:
            self.y[k] = y; self.err[k] = e2
        else:
            self.y[k] += self.alpha * (y - self.y[k])
            self.err[k] += self.alpha * (e2 - self.err[k])
        self.n[k] += 1; self.last[k] = t
    def uncertainty(self, t):
        U = (self.err + 1e-3) * (t - self.last).astype(float)
        U[self.n == 0] = np.inf
        return U


def quantize(u):
    return int(np.argmin(np.abs(LEVELS - np.clip(u, 0.0, 1.0))))


def choose(policy, t, rng, table, hist, coupler, last_sum):
    if policy == "random":
        return int(rng.integers(K))
    if policy == "sweep":
        return t % K
    U = table.uncertainty(t)
    if policy == "curious":
        best = np.flatnonzero(U == U.max())
        return int(rng.choice(best))
    if policy == "coupler":
        return quantize(coupler.step(last_sum))
    if policy == "dual":
        Un = np.where(np.isinf(U), 1.0, U / (U[np.isfinite(U)].max() if np.isfinite(U).any() else 1.0))
        s_hat = table.predict().sum(1)
        h = np.array(list(hist)[-(WINDOW - 1):]) if len(hist) else np.zeros(0)
        var_if = np.array([np.var(np.append(h, s)) for s in s_hat]) if len(h) > 10 else np.full(K, TARGET_VAR)
        score = ((var_if - TARGET_VAR) / TARGET_VAR) ** 2 - 1.0 * Un
        best = np.flatnonzero(score == score.min())
        return int(rng.choice(best))
    raise ValueError(policy)


def run(medium, policy, ticks=2000, seed=0, every=100):
    from .media import response_map
    rng = np.random.default_rng(seed)
    table, coupler = Table(), Coupler()
    hist = deque(maxlen=WINDOW)
    map0 = response_map(medium, LEVELS, rested=True)
    W0 = medium.params().copy()
    used = np.zeros(K, int)
    last_sum = 0.0
    errs, homeo = [], []
    for t in range(ticks):
        k = choose(policy, t, rng, table, hist, coupler, last_sum)
        y = medium.step(LEVELS[k])
        table.update(k, y, t)
        used[k] += 1
        last_sum = float(y.sum()); hist.append(last_sum)
        if len(hist) == WINDOW:
            homeo.append(abs(np.var(hist) - TARGET_VAR))
        if (t + 1) % every == 0:
            truth = response_map(medium, LEVELS)
            errs.append(np.mean(np.abs(table.predict() - truth)))
    map1 = response_map(medium, LEVELS, rested=True)
    p = used / used.sum()
    W1 = medium.params()
    return dict(
        err_final=errs[-1],
        err_late=float(np.mean(errs[len(errs) // 2:])),
        coverage=int((used > 0).sum()),
        entropy_bits=float(-(p[p > 0] * np.log2(p[p > 0])).sum()),
        homeo=float(np.mean(homeo)) if homeo else np.nan,
        drift=float(np.linalg.norm(W1 - W0) / (np.linalg.norm(W0) + 1e-12)),
        spread0=float(map0.std(0).sum()), spread1=float(map1.std(0).sum()),
        usage=used, map0=map0, map1=map1,
    )
