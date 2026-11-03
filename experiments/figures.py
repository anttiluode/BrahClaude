"""Figures for the README, drawn from results/*.json and the model itself."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import FIGS, RESULTS
from brah.edges import TYPES, kernel, taus

S = {"FAC": "#2a78d6", "DEP": "#eb6834", "BAND": "#1baf7a", "AUG": "#eda100"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "font.size": 10, "lines.linewidth": 2,
})
load = lambda n: json.load(open(os.path.join(RESULTS, n + ".json")))


def fig_kernels():
    s = np.arange(1, 401)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for n, g in TYPES.items():
        ax.plot(s, kernel(g, 400), color=S[n], label=n)
    for t in taus():
        ax.axvline(t, color=GRID, lw=1, zorder=0)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("age of an earlier pulse on the route (steps; grey lines = octave band edges)")
    ax.set_ylabel("kappa(age): change in next pulse's gain")
    ax.set_title("What one earlier pulse does to the next one, per edge type", color=INK, loc="left")
    ax.legend(frameon=False, ncol=4, loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "kernels.png"), dpi=150); plt.close(fig)


def fig_g2b():
    r = load("g2")["G2b"]
    cls = r["classes"]; v = np.array(r["homo_scalar_per_class"])
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ages = [20, 40, 80, 160]
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    for i, n in enumerate((1, 2, 3, 4)):
        vals = [v[cls.index([n, d])] for d in ages]
        ax.plot(ages, vals, "o-", color=cols[i], ms=6, label=f"{n} pulse{'s' if n > 1 else ''}")
    ax.set_xscale("log"); ax.set_xticks(ages); ax.set_xticklabels(ages)
    ax.set_xlabel("age of the written burst at ping time (steps)")
    ax.set_ylabel("ping response, homogeneous terminal")
    ax.set_title(f"One band projection confounds amount with age  "
                 f"(homo acc {r['homo_accuracy']:.2f}, octave-tuned acc {r['hetero_accuracy']:.2f})",
                 color=INK, loc="left", fontsize=9.5)
    ax.minorticks_off()
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "g2b_amount_vs_age.png"), dpi=150); plt.close(fig)


def fig_g3():
    r = load("g3")
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2), sharey=False)
    for ax, W in zip(axes, (4, 32, 128)):
        c = r[f"BAND_W{W}"]["learned"]
        lag = np.arange(1, 61)
        k = kernel(TYPES["BAND"], 200)
        ek = np.array([k[np.clip(L + np.arange(W) - 1, 0, 199)].mean() for L in lag])
        ax.plot(lag, ek, color=S["BAND"], label="E[kappa] over arrival window")
        leads = 200 - np.array(c["on_A"])
        ax.plot(leads, np.interp(leads, lag, ek), "o", color=INK, ms=7, label="learned writes")
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_title(f"W = {W}  (mean lead {c['mean_lead']:.1f})", color=INK, fontsize=10)
        ax.set_xlabel("lead before t0 (steps)")
    axes[0].set_ylabel("expected gain added at arrival")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("BAND route: a reward-only learner writes where the residue will peak when the signal arrives",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "g3_write_for_later.png"), dpi=150); plt.close(fig)


def fig_g4():
    import g4_multiplex as g
    rng = np.random.default_rng(0)
    T = 60_000
    a = 0.005 + 0.025 * g.sig(1.5 * g.ou(rng, T, g.TAU_SIG))
    b = 0.1 + 0.8 * g.sig(1.5 * g.ou(rng, T, g.TAU_SIG))
    x = np.zeros(T); t, last = 0, -99
    while t < T - 6:
        if t - last > g.REFR and rng.random() < a[t]:
            n = 3 if rng.random() < b[t] else 1
            x[t + 2 * np.arange(n)] = 1; last = t + 2 * (n - 1); t = last + 1
        else:
            t += 1
    from brah.edges import run
    y, _ = run(np.tile(x[:, None], (1, 3)), np.stack([g.GD, g.GF, np.zeros(8)]))
    lp = g.lowpass(y, g.TAU_LP)[2500:]; ar = g.lowpass(a, g.TAU_LP)[2500:]; br = g.lowpass(b, g.TAU_LP)[2500:]
    z = lambda u: (u - u.mean()) / u.std()
    tt = np.arange(len(ar))
    fig, axes = plt.subplots(3, 1, figsize=(10, 5.6), sharex=True)
    axes[0].plot(tt, z(ar), color=INK2, lw=1.5, label="a(t) event rate")
    axes[0].plot(tt, z(lp[:, 0]), color=S["DEP"], label="depressing target")
    axes[0].set_title("Depressing route reads the event rate", loc="left", fontsize=10, color=INK)
    axes[1].plot(tt, z(br), color=INK2, lw=1.5, label="b(t) burst probability")
    axes[1].plot(tt, z(lp[:, 1] / lp[:, 0]), color=S["FAC"], label="facilitating / depressing")
    axes[1].set_title("Facilitating route over depressing route reads burstiness", loc="left", fontsize=10, color=INK)
    axes[2].plot(tt, z(ar), color=INK2, lw=1.2, label="a(t)")
    axes[2].plot(tt, z(br), color="#e87ba4", lw=1.2, label="b(t)")
    axes[2].plot(tt, z(lp[:, 2]), color=INK, lw=1.5, label="residue-free route (mixes both)")
    axes[2].set_title("Without residue the two messages arrive mixed", loc="left", fontsize=10, color=INK)
    for ax in axes:
        ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0)); ax.set_ylabel("z-score")
    axes[2].set_xlabel("time (steps)")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "g4_multiplex.png"), dpi=150); plt.close(fig)


if __name__ == "__main__":
    fig_kernels(); fig_g2b(); fig_g3(); fig_g4()
    print("figures written")
