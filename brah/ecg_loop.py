"""
Gate 0 - a faithful, Qt-free port of the original PerceptionLab ecg.json loop.

    Constant(1.0) -> setpoint_mod
    Coupler.signal_out -> Checkerboard.square_size
    Checkerboard.image -> ImageToVector -> VectorSplitter
    VectorSplitter.out_0..out_3 -> Coupler.signal_in   (SUMMED by get_blended_input 'sum')

Every node body below is copied from the node files in
anttiluode/GeometricNeuronOriginReview, with the Qt display code removed.
The only thing that is NOT in those files is the scheduler, which lives in
PerceptionLab's main loop. Two schedules are offered:

  'sequential'  nodes step in id order, each reading the freshest output
                (coupler reads the splitter from the previous tick -> loop delay 1)
  'synchronous' every node reads the previous tick's outputs (loop delay 4)
"""
from collections import deque
import numpy as np
import cv2


def checkerboard(size_in, size=256):
    size_in = size_in or 0.1                      # original: `or 0.1`
    square = int(5 + size_in * 50)                # original: no clipping
    if square == 0:                               # numpy would warn and give 0s
        square = 1
    y, x = np.mgrid[0:size, 0:size]
    return (((x // square) + (y // square)) % 2).astype(np.float32), square


def image_to_vector(img, output_dim):
    side = int(np.ceil(np.sqrt(output_dim)))
    tiny = cv2.resize(img, (side, side), interpolation=cv2.INTER_AREA)
    flat = tiny.flatten()
    if len(flat) >= output_dim:
        v = flat[:output_dim].copy()
    else:
        v = np.zeros(output_dim, np.float32); v[:len(flat)] = flat
    m = np.max(np.abs(v))
    return v / m if m > 0 else v


class Coupler:
    """HomeostaticCouplerNode, edge_of_chaos mode, parameters from ecg.json."""
    def __init__(self, setpoint=0.5, sharpness=3.0, target_variance=0.1,
                 variance_window=50, gain=1.0):
        self.setpoint, self.sharpness = setpoint, sharpness
        self.target_variance, self.variance_window = target_variance, variance_window
        self.gain = gain
        self.history = deque(maxlen=200)
        self.integral_state = 0.0
        self.output_value = 0.0
        self.regime_value = 0.0

    def _sigmoid(self, x, center, sharpness):
        z = (x - center) * sharpness
        return 2.0 / (1.0 + np.exp(-z)) - 1.0

    def step(self, sig_in, sp_mod=1.0):
        eff_sp = self.setpoint + float(sp_mod) * 0.1
        value = float(sig_in)
        self.history.append(value)
        if len(self.history) > 10:
            cur_var = np.var(np.array(self.history)[-self.variance_window:])
        else:
            cur_var = 0.0
        var_err = cur_var - self.target_variance
        err = value - eff_sp
        if var_err > 0:
            out = eff_sp + self._sigmoid(err, 0.0, self.sharpness) / self.sharpness
            self.regime_value = -1.0
        else:
            amp = eff_sp + err * (1.0 + abs(var_err) * 10.0)
            out = eff_sp + np.tanh((amp - eff_sp) * 2.0)
            self.regime_value = 1.0
        if len(self.history) > 20:   # computed but unused in edge mode (kept for fidelity)
            self.integral_state = self.integral_state * 0.99 + np.mean(list(self.history)[-50:]) * 0.01
        self.output_value = out * self.gain
        return self.output_value


def run_loop(ticks=1500, output_dim=256, variance_window=50, schedule="sequential",
             n_taps=4, init_taps=(0.0, 0.0, 0.375, 1.0), init_out=0.8739509815096245):
    """Returns dict of per-tick series: coupler input, coupler output, square size, regime."""
    cp = Coupler(variance_window=variance_window)
    taps = np.array(init_taps[:n_taps], float)     # splitter state saved in ecg.json
    cp_out = init_out                              # coupler output saved in ecg.json
    rec = {k: [] for k in ("sig_in", "out", "square", "regime")}
    for _ in range(ticks):
        if schedule == "sequential":
            sig = taps.sum()
            cp_out = cp.step(sig)
            img, sq = checkerboard(cp_out)
            taps = image_to_vector(img, output_dim)[:n_taps]
        else:  # synchronous: 4-stage pipeline, each stage reads last tick's value
            sig = taps.sum()
            new_out = cp.step(sig)
            img, sq = checkerboard(cp_out)
            new_taps = image_to_vector(img, output_dim)[:n_taps]
            cp_out, taps = new_out, new_taps
        rec["sig_in"].append(sig); rec["out"].append(cp_out)
        rec["square"].append(sq); rec["regime"].append(cp.regime_value)
    return {k: np.array(v) for k, v in rec.items()}


def dominant_period(x, burn=300, max_lag=200):
    """Lag of the highest autocorrelation peak after the first zero crossing."""
    x = np.asarray(x[burn:], float); x = x - x.mean()
    if x.std() < 1e-12:
        return 0, 1.0
    ac = np.correlate(x, x, "full")[len(x) - 1:][:max_lag + 1]
    ac = ac / ac[0]
    neg = np.where(ac < 0)[0]
    if len(neg) == 0:
        return 0, 1.0
    start = neg[0]
    lag = start + int(np.argmax(ac[start:]))
    return lag, float(ac[lag])


def distinguishability(output_dim, sizes=range(5, 60), n_taps=4):
    """Sol's static test: how many square sizes give distinct 4-tap vectors / sums."""
    vecs, sums = [], []
    for s in sizes:
        yy, xx = np.indices((256, 256))
        pat = ((xx // s + yy // s) % 2).astype(np.float32)
        v = image_to_vector(pat, output_dim)[:n_taps]
        vecs.append(tuple(np.round(v, 6))); sums.append(round(float(v.sum()), 6))
    return len(set(vecs)), len(set(sums))
