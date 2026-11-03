import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brah.edges import TYPES, kernel, lams, response_at, run  # noqa: E402

G = np.stack([TYPES[n] for n in ["FAC", "DEP", "BAND", "AUG"]])


def test_telescoping():
    rng = np.random.default_rng(0)
    x = (rng.random((500, 4)) < 0.1).astype(float)
    _, _, H = run(x, G, clip=False, return_bands=True)
    m = np.zeros((4, 8))
    for t in range(500):
        assert np.allclose(H[t].sum(-1), m[:, -1], atol=1e-12)
        m = lams() * m + x[t][:, None]


def test_volterra():
    rng = np.random.default_rng(1)
    x = (rng.random((400, 4)) < 0.1).astype(float)
    y, _ = run(x, G, clip=False)
    k = kernel(G, 400)
    for e in range(4):
        conv = np.convolve(x[:, e], np.concatenate([[0.0], k[e]]))[:400]
        assert np.allclose(y[:, e], x[:, e] * (1 + conv), atol=1e-12)


def test_first_pulse_blank():
    assert np.all(response_at([0], None, G) == 1.0)


def test_silence_is_silent():
    x = np.zeros((300, 4)); x[:3] = 1
    y, m = run(x, G)
    assert np.all(y[3:] == 0) and np.all(m > 0)
