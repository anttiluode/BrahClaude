"""G0 — the model identities (telescoping, Volterra form, first pulse is blank)."""
import numpy as np

from common import save
from brah.edges import TYPES, bands_from_traces, kernel, lams, run

rng = np.random.default_rng(0)
T, E, K = 2000, 4, 8
x = (rng.random((T, E)) < 0.08).astype(float)
G = np.stack([TYPES[n] for n in ["FAC", "DEP", "BAND", "AUG"]])

# G0a telescoping: sum of bands == slowest trace, at every step
y, m, H = run(x, G, clip=False, return_bands=True)
lam = lams(K)
mm = np.zeros((E, K))
tele_err = 0.0
for t in range(T):
    tele_err = max(tele_err, np.abs(H[t].sum(-1) - mm[:, -1]).max())
    mm = lam * mm + x[t][:, None]

# G0b unclipped edge == explicit Volterra with kappa
kap = kernel(G, T)  # (E, T) kappa(s=1..T)
yv = np.zeros_like(y)
for e in range(E):
    conv = np.convolve(x[:, e], np.concatenate([[0.0], kap[e]]))[:T]
    yv[:, e] = x[:, e] * (1.0 + conv)
volt_err = float(np.abs(y - yv).max())

# G0c first pulse on a fresh edge is exactly 1 for every type
xf = np.zeros((1, E)); xf[0] = 1
y1, _ = run(xf, G)
first = y1[0].tolist()

res = {
    "G0a_telescoping_max_err": tele_err,
    "G0b_volterra_max_err": volt_err,
    "G0c_first_pulse": first,
    "pass": {
        "G0a": tele_err < 1e-9,
        "G0b": volt_err < 1e-9,
        "G0c": all(v == 1.0 for v in first),
    },
}
print(res)
save("g0", res)
