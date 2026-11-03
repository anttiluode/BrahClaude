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


def test_neutral_read_is_common_mode_on_linear_route():
    # a pulse on every route shifts all answers equally (before the clip)
    x = np.zeros((60, 4)); x[[0, 2, 4], 0] = 1; x[[10, 20, 30], :] = 1; x[50] = 1
    x0 = x.copy(); x0[[10, 20, 30], :] = 0
    gA = np.tile(TYPES["AUG"], (4, 1))
    y, _ = run(x, gA, clip=False); y0, _ = run(x0, gA, clip=False)
    assert np.allclose(y[50] - y[50, 1], y0[50] - y0[50, 1], atol=1e-12)


def test_tm_fresh_and_saturating():
    from brah import tm
    y, _ = tm.run(np.ones((1, 3)))
    assert np.allclose(y, 1.0, atol=1e-12)
    x = np.zeros((400, 2)); x[[0, 2, 4], 0] = 1; x[200:, :] = 1   # then hammer both
    y, _ = tm.run(x)
    assert abs(y[-1, 0] - y[-1, 1]) < 1e-6                       # dense reads erase the difference
