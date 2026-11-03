"""
Gate 2B - the one question the 10-seed table cannot settle:
does an adaptive medium (fatigue, plasticity) push the UNCHANGED homeostatic
coupler into exploring more pings than the same network without adaptation?

Paired design: seed s builds the SAME network wiring for all three media,
so each seed gives a within-pair difference.
"""
import os, sys  # run from anywhere: repo root on path, outputs land in pinger/
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(_HERE), _HERE]
os.chdir(_HERE)
import json, os
import numpy as np
from brah.media import Network
from brah import pinger

N = int(os.environ.get("BRAH_PAIRED", 60))
rows = {"static-net": [], "habituating": [], "plastic": []}
for s in range(N):
    for kind in rows:
        r = pinger.run(Network(s, adapt=(kind != "static-net"), plastic=(kind == "plastic")),
                       "coupler", 2000, seed=s, every=500)
        rows[kind].append((r["coverage"], r["err_late"], r["homeo"]))
res = {}
base = np.array(rows["static-net"])
for kind, v in rows.items():
    v = np.array(v)
    d = v[:, 0] - base[:, 0]
    res[kind] = {
        "coverage_mean": round(float(v[:, 0].mean()), 2),
        "coverage_median": float(np.median(v[:, 0])),
        "frac_seeds_locked_<=4_pings": round(float((v[:, 0] <= 4).mean()), 3),
        "err_late_mean": round(float(v[:, 1].mean()), 4),
        "homeo_mean": round(float(v[:, 2].mean()), 4),
        "paired_cov_diff_vs_static": round(float(d.mean()), 2),
        "paired_diff_se": round(float(d.std(ddof=1) / np.sqrt(N)), 2) if kind != "static-net" else 0.0,
        "seeds_more_coverage_than_static": int((d > 0).sum()),
        "seeds_less_coverage_than_static": int((d < 0).sum()),
    }
    print(kind, res[kind], flush=True)
os.makedirs("results", exist_ok=True)
json.dump({"n_pairs": N, "ticks": 2000, "result": res}, open("results/GATE2B_paired.json", "w"), indent=1)
