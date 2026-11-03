"""Temporal-Sihti edges: a route that keeps its own pulse history as octave bands.

Each edge keeps K unnormalised exponential traces of the pulses that crossed it,

    m_k(t+1) = lam_k * m_k(t) + x(t),      lam_k = exp(-1/tau_k),  tau_k = 2, 4, ..., 2^K

(read *before* the update, so a pulse at time t sees only the past).  The Sihti
residue stack of that history is

    h_0 = m_0,   h_k = m_k - m_{k-1}   (k = 1..K-1)

and it telescopes exactly:  sum_k h_k = m_{K-1}.  Band h_k is roughly "how many
pulses crossed this edge between tau_{k-1} and tau_k steps ago" -- octave age bins.

A pulse leaving the edge is

    y(t) = w * x(t) * max(0, 1 + g . h(t))

so the incoming pulse is binary and stereotyped; what it *carries* out is set by
the edge's band gains g and the residue left by earlier pulses.  Positive g on fast
bands = facilitation, negative = depression, gains on slow bands = augmentation.

Before the clip this is a second-order Volterra system (not LTI):

    y(t) = w x(t) [1 + sum_{s>=1} kappa(s) x(t-s)],   kappa(s) = sum_k G_k lam_k^(s-1),
    G_k = g_k - g_{k+1}  (g_K := 0)

so the band basis is a reparametrisation of a sum-of-exponentials kernel.  That is
checked numerically in experiments/g0_identities.py.
"""
from __future__ import annotations

import numpy as np

K_DEFAULT = 8


def taus(K: int = K_DEFAULT) -> np.ndarray:
    return 2.0 ** np.arange(1, K + 1)


def lams(K: int = K_DEFAULT) -> np.ndarray:
    return np.exp(-1.0 / taus(K))


def bands_from_traces(m: np.ndarray) -> np.ndarray:
    """Sihti residue stack h from traces m (last axis = K)."""
    h = np.empty_like(m)
    h[..., 0] = m[..., 0]
    h[..., 1:] = m[..., 1:] - m[..., :-1]
    return h


def kernel(g: np.ndarray, smax: int, K: int | None = None) -> np.ndarray:
    """kappa(s) for s = 1..smax (index 0 of the result is s=1)."""
    g = np.asarray(g, float)
    K = g.shape[-1] if K is None else K
    lam = lams(K)
    G = g - np.concatenate([g[..., 1:], np.zeros(g.shape[:-1] + (1,))], axis=-1)
    s = np.arange(1, smax + 1)
    return (G[..., None, :] * lam[None, :] ** (s[:, None] - 1)).sum(-1)


def run(x: np.ndarray, g: np.ndarray, w: float | np.ndarray = 1.0,
        m0: np.ndarray | None = None, clip: bool = True,
        noise: float = 0.0, rng: np.random.Generator | None = None,
        return_bands: bool = False):
    """Drive E edges with pulse trains.

    x : (T, E) pulse amplitudes (0/1 for spikes)
    g : (E, K) or (K,) band gains
    returns y (T, E), final traces m (E, K)[, bands (T, E, K)]
    Noise (if any) is added only where a pulse is present (release variability).
    """
    x = np.asarray(x, float)
    T, E = x.shape
    g = np.broadcast_to(np.asarray(g, float), (E, np.shape(g)[-1]))
    K = g.shape[1]
    lam = lams(K)
    m = np.zeros((E, K)) if m0 is None else np.array(m0, float)
    y = np.zeros((T, E))
    H = np.zeros((T, E, K)) if return_bands else None
    for t in range(T):
        h = bands_from_traces(m)
        if return_bands:
            H[t] = h
        gain = 1.0 + (g * h).sum(1)
        if clip:
            gain = np.maximum(gain, 0.0)
        y[t] = w * x[t] * gain
        m = lam[None, :] * m + x[t][:, None]
    if noise > 0:
        rng = rng or np.random.default_rng()
        y = y + (x != 0) * rng.normal(0.0, noise, size=y.shape)
    out = (y, m)
    if return_bands:
        out = out + (H,)
    return out


def response_at(pulse_times, read_times, g, T=None, clip=True):
    """Deterministic responses of one edge (or several, g (E,K)) to a train.

    pulse_times: iterable of ints; returns y at each pulse time, shape (n_pulses, E).
    """
    pt = np.sort(np.asarray(list(pulse_times), int))
    T = int(pt.max()) + 1 if T is None else T
    g = np.atleast_2d(g)
    x = np.zeros((T, g.shape[0]))
    x[pt] = 1.0
    y, _ = run(x, g, clip=clip)
    return y[pt]


# --- edge types used across the gates (gains on bands with ages ~2,4,...,256) ---
TYPES = {
    "FAC":  np.array([0.8, 0.6, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0]),    # fast facilitation
    "DEP":  np.array([-0.6, -0.4, -0.2, -0.1, 0.0, 0.0, 0.0, 0.0]),  # depression
    "BAND": np.array([-0.4, -0.2, 0.3, 0.5, 0.3, 0.0, 0.0, 0.0]),   # depress then facilitate later
    "AUG":  np.array([0.0, 0.0, 0.0, 0.0, 0.2, 0.3, 0.3, 0.2]),     # slow augmentation
}
