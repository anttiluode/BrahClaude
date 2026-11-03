"""
Attacker on verdicts 2-3: was the original coupler only losing on the networks
because its knobs were tuned for the checkerboard?

For each network medium, grid-search the coupler's setpoint, internal target
variance and sharpness on TUNING seeds 100-104, choosing the setting with the
best homeostasis score (the same fixed goal as everywhere: |var - 0.1|).
Then score that retuned coupler on HELD-OUT seeds 0-19 against the default
coupler and against `dual`, all on identical media.
"""
import os, sys  # run from anywhere: repo root on path, outputs land in pinger/
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(_HERE), _HERE]
os.chdir(_HERE)
import itertools, json, os
import numpy as np
from brah.media import Network
from brah import pinger

GRID = {
    "setpoint": [0.1, 0.25, 0.4, 0.55],          # effective setpoint = this + 0.1
    "target_variance": [0.03, 0.1, 0.2, 0.4],
    "sharpness": [1.5, 3.0, 6.0],
}
TUNE, HELD = range(100, 105), range(0, 20)
TICKS = 2000


def med(kind, s):
    return Network(s, adapt=(kind != "static-net"), plastic=(kind == "plastic"))


def score(kind, seeds, policy, kw=None):
    rows = [pinger.run(med(kind, s), policy, TICKS, seed=s, every=500, coupler_kw=kw) for s in seeds]
    return {k: round(float(np.mean([r[k] for r in rows])), 4) for k in ("homeo", "err_late", "coverage")}


if __name__ == "__main__":
    out = {"grid": GRID, "tune_seeds": list(TUNE), "held_out_seeds": list(HELD), "result": {}}
    for kind in ["static-net", "habituating", "plastic"]:
        best, best_kw = None, None
        for vals in itertools.product(*GRID.values()):
            kw = dict(zip(GRID, vals))
            h = score(kind, TUNE, "coupler", kw)["homeo"]
            if best is None or h < best:
                best, best_kw = h, kw
        res = {
            "best_kw": best_kw, "tune_homeo": best,
            "default_coupler": score(kind, HELD, "coupler"),
            "retuned_coupler": score(kind, HELD, "coupler", best_kw),
            "dual": score(kind, HELD, "dual"),
        }
        out["result"][kind] = res
        print(kind, json.dumps(res), flush=True)
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/ATTACKER_retuned_coupler.json", "w"), indent=1)
