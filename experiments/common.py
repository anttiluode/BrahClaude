import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def save(name, obj):
    path = os.path.join(RESULTS, name + ".json")
    with open(path, "w") as f:
        json.dump(_clean(obj), f, indent=2)
    return path


def bayes_accuracy(templates, sigma, n_trials, rng):
    """Equiprobable classes, isotropic Gaussian noise -> nearest template is Bayes.

    templates: (C, D). Returns accuracy estimated with fresh noise.
    """
    templates = np.asarray(templates, float)
    C, D = templates.shape
    labels = rng.integers(0, C, n_trials)
    obs = templates[labels] + rng.normal(0, sigma, (n_trials, D))
    d2 = ((obs[:, None, :] - templates[None, :, :]) ** 2).sum(-1)
    return float((d2.argmin(1) == labels).mean())
