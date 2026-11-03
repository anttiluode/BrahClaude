"""
BrahClaude gate runner.  `python run_gates.py` -> results/GATES.json + printed tables.
Pure numpy + opencv-python (cv2 only for the original INTER_AREA resize).
"""
import json, os, sys, time
import numpy as np
from brah.ecg_loop import run_loop, dominant_period, distinguishability
from brah.media import Checkerboard, Network
from brah import pinger

SEEDS = int(os.environ.get("BRAH_SEEDS", 10))
TICKS = int(os.environ.get("BRAH_TICKS", 2000))
POLICIES = ["coupler", "random", "sweep", "curious", "dual"]
out = {"seeds": SEEDS, "ticks": TICKS}


def gate0():
    g = {"distinguishability": {}, "rhythm": {}}
    for d in (256, 1024, 2048):
        v, s = distinguishability(d)
        g["distinguishability"][d] = {"distinct_4tap": v, "distinct_sum": s}
    for sched in ("sequential", "synchronous"):
        for d in (256, 1024, 2048):
            for w in (50, 30):
                r = run_loop(1500, d, w, sched)
                p, a = dominant_period(r["out"])
                g["rhythm"][f"{sched}/{d}/w{w}"] = {
                    "period": int(p), "autocorr": round(a, 3),
                    "square_sizes_visited": int(len(set(r["square"][300:]))),
                }
    return g


def make(kind, seed):
    if kind == "checkerboard":
        return Checkerboard(clip=True)
    return Network(seed, adapt=(kind != "static-net"), plastic=(kind == "plastic"))


def gate_media(kinds):
    res = {}
    for kind in kinds:
        res[kind] = {}
        for pol in POLICIES:
            rows = [pinger.run(make(kind, s), pol, TICKS, seed=s) for s in range(SEEDS)]
            agg = {}
            for key in ("err_final", "err_late", "coverage", "entropy_bits", "homeo",
                        "drift", "spread0", "spread1"):
                vals = np.array([r[key] for r in rows], float)
                agg[key] = [round(float(vals.mean()), 4), round(float(vals.std()), 4)]
            # does the medium end up shaped by the pinger's habits?
            cors = []
            for r in rows:
                dg = np.linalg.norm(r["map1"], axis=1) - np.linalg.norm(r["map0"], axis=1)
                if r["usage"].std() > 0 and dg.std() > 1e-9:
                    cors.append(np.corrcoef(r["usage"], dg)[0, 1])
            agg["usage_vs_gain_change_r"] = [round(float(np.mean(cors)), 3), len(cors)] if cors else [None, 0]
            res[kind][pol] = agg
            print(f"{kind:12s} {pol:8s} err_late {agg['err_late'][0]:.3f}±{agg['err_late'][1]:.3f} "
                  f"cov {agg['coverage'][0]:5.1f} H {agg['entropy_bits'][0]:.2f}b "
                  f"homeo {agg['homeo'][0]:.3f} drift {agg['drift'][0]:.3f} "
                  f"spread {agg['spread0'][0]:.3f}->{agg['spread1'][0]:.3f} "
                  f"r(use,dgain) {agg['usage_vs_gain_change_r'][0]}", flush=True)
    return res


if __name__ == "__main__":
    t0 = time.time()
    out["gate0"] = gate0()
    print(json.dumps(out["gate0"], indent=1))
    out["gates123"] = gate_media(["checkerboard", "static-net", "habituating", "plastic"])
    out["runtime_s"] = round(time.time() - t0, 1)
    os.makedirs("results", exist_ok=True)
    with open("results/GATES.json", "w") as f:
        json.dump(out, f, indent=1)
    print("runtime", out["runtime_s"], "s")
