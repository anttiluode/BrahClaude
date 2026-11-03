"""Tsodyks-Markram short-term plasticity route (discrete time, one step ~ 5 ms).

Facilitation u (utilisation) and resources R per route:

    between spikes:  u <- U + (u - U) exp(-1/tau_F)      R <- 1 - (1 - R) exp(-1/tau_D)
    on a spike:      u <- u + U (1 - u)                   (facilitate first)
                     y  = u R / y0                        (release)
                     R <- R - u R                         (deplete)

y0 = 2U - U^2 is the release of a fresh synapse, so a fresh route answers exactly 1,
as the temporal-Sihti edge does (G0c).  Defaults follow the proportions of Mongillo,
Barak & Tsodyks 2008 (U = 0.2, tau_F = 1.5 s, tau_D = 0.2 s) at 5 ms per step.

Unlike brah/edges.py this route is multiplicative and saturating: u is bounded by 1
and every spike spends R.  A neutral ping on every route is therefore NOT a
common-mode write - it pushes all routes toward the same saturated state.
"""
from __future__ import annotations

import numpy as np

U_DEF, TAU_F_DEF, TAU_D_DEF = 0.2, 300.0, 40.0


def run(x: np.ndarray, U: float = U_DEF, tau_f: float = TAU_F_DEF,
        tau_d: float = TAU_D_DEF):
    """x: (T, E) 0/1 spikes. Returns y (T, E) normalised releases and final (u, R)."""
    x = np.asarray(x, float)
    T, E = x.shape
    df, dd = np.exp(-1.0 / tau_f), np.exp(-1.0 / tau_d)
    y0 = 2 * U - U * U
    u = np.full(E, U)
    R = np.ones(E)
    y = np.zeros((T, E))
    for t in range(T):
        s = x[t]
        u_sp = u + U * (1.0 - u) * s
        rel = u_sp * R * s
        y[t] = rel / y0
        R = R - rel
        u = u_sp
        # relax toward rest before the next step
        u = U + (u - U) * df
        R = 1.0 - (1.0 - R) * dd
    return y, (u, R)
