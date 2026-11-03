"""
The things that sit where the checkerboard sat.

Every medium takes a scalar ping u (the coupler's signal_out) and returns a
4-vector (the four splitter taps).  Every medium can be CLONED so its current
response map can be measured without the measurement disturbing the run -
a privilege a real neuron does not have, used here only for scoring.

  Checkerboard   the original: fixed geometry, no state.
  Habituating    48 rate neurons, fixed wiring, each neuron fatigues when it
                 fires and recovers when it rests. History-dependent, nothing
                 permanent.
  Plastic        the same, plus Hebbian (Oja) learning on the recurrent
                 weights. Being pinged permanently rewires it.
"""
import copy
import numpy as np
from .ecg_loop import checkerboard, image_to_vector


class Checkerboard:
    kind = "checkerboard"
    def __init__(self, output_dim=256, clip=False):
        self.output_dim, self.clip = output_dim, clip
    def step(self, u):
        if self.clip:
            u = float(np.clip(u, 0.0, 1.0))
        img, _ = checkerboard(u)
        return image_to_vector(img, self.output_dim)[:4].astype(float)
    def clone(self):
        return copy.deepcopy(self)
    def params(self):
        return np.zeros(1)


class Network:
    """
    r <- tanh(relu(B*I(u) + W r - a))   (3 settling substeps per tick)
    I(u)  : Gaussian tuning curves, random preferred pings in [0,1]
    taps  : neurons 0..3, which get NO direct input - they only hear the
            network, so the return is never just the tuning curve.
    a     : adaptation (fatigue), tau_a ticks
    W     : fixed random recurrent wiring
    plastic=True : preferred pings (centers) drift toward the pings heard
    """
    def __init__(self, seed, n=48, sigma=0.06, tau_a=20.0, kappa=1.0,
                 plastic=False, eta=0.02, adapt=True):
        rng = np.random.default_rng(seed)
        self.kind = ("plastic" if plastic else "habituating") if adapt else "static-net"
        self.n, self.sigma, self.tau_a, self.kappa = n, sigma, tau_a, kappa
        self.plastic, self.eta, self.adapt = plastic, eta, adapt
        self.centers = rng.uniform(0, 1, n)
        self.B = np.abs(rng.normal(1.0, 0.3, n)); self.B[:4] = 0.0
        W = rng.normal(0, 0.9 / np.sqrt(n), (n, n)); np.fill_diagonal(W, 0)
        W[:4] = rng.normal(0, 5.0 / np.sqrt(n), (4, n))   # taps: mixed-sign listeners
        W[:4, :4] = 0.0
        self.W = W
        self.row_norm = np.linalg.norm(W, axis=1)
        self.W0 = W.copy()
        self.r = np.zeros(n); self.a = np.zeros(n)

    def drive(self, u):
        return self.B * np.exp(-(u - self.centers) ** 2 / (2 * self.sigma ** 2))

    def step(self, u):
        u = float(np.clip(u, 0.0, 1.0))
        I = self.drive(u)
        r = self.r
        for _ in range(3):
            r = np.tanh(np.maximum(0.0, I + self.W @ r - self.a))
        if self.adapt:
            self.a += (self.kappa * r - self.a) / self.tau_a
        if self.plastic:
            # Map plasticity: each input neuron's preferred ping moves toward
            # the pings that make it fire (competitive / Kohonen-style).
            # Frequently used pings recruit more neurons; unused ones lose them.
            # (Oja/BCM on the recurrent W was tried first and collapsed the
            #  network into an input-blind attractor at every eta tried.)
            self.centers += self.eta * r * (u - self.centers)
            self.centers[:4] = 0.0
        self.r = r
        return r[:4].copy()

    def clone(self):
        return copy.deepcopy(self)

    def params(self):
        return self.centers.copy()


def response_map(medium, pings, rested=False):
    """The medium's CURRENT answer to every ping, each asked of a fresh clone.
    rested=True first clears fast state (activity, fatigue) on the clone, so
    only the PERMANENT part of the medium (its wiring / tuning) is measured."""
    out = []
    for u in pings:
        c = medium.clone()
        if rested and hasattr(c, "a"):
            c.a[:] = 0.0; c.r[:] = 0.0
        out.append(c.step(u))
    return np.array(out)
