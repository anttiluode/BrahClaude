"""G2 — silent memory on routes, read by a neutral ping (Wolff et al. 2017 style).

G2a: what was written (which routes) after a silent delay.
G2b: amount vs age — one projection confounds them, octave-tuned terminals separate them.
"""
import itertools

import numpy as np

from common import bayes_accuracy, save
from brah.edges import TYPES, run

rng = np.random.default_rng(2)


def burst(T, start, n, isi=2):
    x = np.zeros(T)
    x[start + isi * np.arange(n)] = 1.0
    return x


# ---------------- G2a ----------------
E, D, SIG_A = 8, 150, 0.1
pairs = list(itertools.combinations(range(E), 2))
items = [pairs[i] for i in rng.choice(len(pairs), 8, replace=False)]
T = 5 + D + 1
t_ping = T - 1
G_aug = np.tile(TYPES["AUG"], (E, 1))

templ, delay_out, finals = [], [], []
for it in items:
    x = np.zeros((T, E))
    for r in it:
        x[:, r] = burst(T, 0, 3)
    x[t_ping] = 1.0
    y, _ = run(x, G_aug)
    templ.append(y[t_ping])
    delay_out.append(float(np.abs(y[6:t_ping]).sum()))
    # traces just before the ping (for the reset control)
    _, m = run(x[:t_ping], G_aug)
    finals.append(m)
templ = np.array(templ)
acc_ping = bayes_accuracy(templ, SIG_A, 20000, rng)

# reset control: each route's residue replaced by the across-route mean before the ping
templ_reset = []
for m in finals:
    m_mean = np.tile(m.mean(0), (E, 1))
    xp = np.ones((1, E))
    y, _ = run(xp, G_aug, m0=m_mean)
    templ_reset.append(y[0])
templ_reset = np.array(templ_reset)
# decode reset observations against the *intact* templates (the reader's knowledge)
labels = rng.integers(0, 8, 20000)
obs = templ_reset[labels] + rng.normal(0, SIG_A, (20000, E))
acc_reset = float((((obs[:, None] - templ[None]) ** 2).sum(-1).argmin(1) == labels).mean())

g2a = {
    "items": items,
    "ping_accuracy": acc_ping,
    "delay_total_output": max(delay_out),
    "reset_accuracy": acc_reset,
    "ping_templates_active_vs_inactive": [float(templ[0][list(items[0])].mean()),
                                          float(np.delete(templ[0], list(items[0])).mean())],
    "pass": acc_ping >= 0.90 and max(delay_out) == 0.0 and acc_reset <= 0.175,
}
print("G2a", g2a)

# ---------------- G2b ----------------
SIG_B = 0.05
hetero = np.zeros((4, 8))
for i, (a, b) in enumerate([(3, 4), (4, 5), (5, 6), (6, 7)]):
    hetero[i, [a, b]] = 0.3
homo = np.tile(hetero.mean(0), (4, 1))
classes = [(n, d) for n in (1, 2, 3, 4) for d in (20, 40, 80, 160)]


def ping_templates(G):
    out = []
    for n, d in classes:
        Tb = 2 * n + d + 1
        x1 = burst(Tb, 0, n)
        t_ping = 2 * (n - 1) + d  # ping D steps after the last burst spike
        x1[t_ping] = 1.0
        x = np.tile(x1[: t_ping + 1, None], (1, 4))
        y, _ = run(x, G)
        out.append(y[t_ping])
    return np.array(out)


Th, To = ping_templates(hetero), ping_templates(homo)
acc_h = bayes_accuracy(Th, SIG_B, 20000, rng)
acc_o = bayes_accuracy(To, SIG_B, 20000, rng)
g2b = {
    "classes": classes,
    "hetero_accuracy": acc_h,
    "homo_accuracy": acc_o,
    "homo_scalar_per_class": To[:, 0].round(4),
    "pass": acc_h >= 0.80 and acc_o <= 0.50,
}
print("G2b", g2b)
save("g2", {"G2a": g2a, "G2b": g2b})
