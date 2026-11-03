"""
Gate 4 - "it would learn to ping what it wants"

Each EPISODE the pinger is handed a target return y* (a 4-tap pattern) and
must make the medium produce it: ||y - y*|| < TOL.  It pings until it hits
or gives up after K=32 pings.  The medium persists across episodes, so a
pinger that remembers can get faster, and a living medium can drift, tire,
or rewire under the goal-directed pinging.

Targets are always reachable at the moment they are set: y* is the medium's
current answer to a randomly chosen ping, read off a clone.

Policies
  blind    random pings, no repeats within an episode (the smart blind
           baseline; expected ~K/(hits+1) pings)
  ashby    Ashby's homeostat as a goal-seeker: a random jump on every miss,
           repeats allowed, no memory (ultrastability, 1952)
  memory   inverse model from the Gate 1-3 memory table: try pings in order
           of how close their remembered return is to y*; every ping, hit or
           miss, updates the table; the table persists across episodes
  oracle   the same ordering but read off the TRUE current map (a clone).
           Not a policy a neuron could run - the floor.
"""
import os, sys  # run from anywhere: repo root on path, outputs land in pinger/
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(_HERE), _HERE]
os.chdir(_HERE)
import json, os
import numpy as np
from brah.media import Checkerboard, Network, response_map
from brah.pinger import LEVELS, K, Table

TOL = 0.15
EPISODES = int(os.environ.get("BRAH_EPISODES", 300))
SEEDS = int(os.environ.get("BRAH_SEEDS", 20))
POLICIES = ["blind", "ashby", "memory", "oracle"]
MEDIA = ["checkerboard", "static-net", "habituating", "plastic"]


def make(kind, seed):
    if kind == "checkerboard":
        return Checkerboard(clip=True)
    return Network(seed, adapt=(kind != "static-net"), plastic=(kind == "plastic"))


def episode(medium, policy, table, ystar, rng, t):
    tried = np.zeros(K, bool)
    order = None
    if policy == "memory":
        order = list(np.argsort(np.linalg.norm(table.predict() - ystar, axis=1)))
    for step in range(1, K + 1):
        if policy == "blind":
            k = int(rng.choice(np.flatnonzero(~tried)))
        elif policy == "ashby":
            k = int(rng.integers(K))
        elif policy == "memory":
            # re-rank after every miss: the table just learned something
            d = np.linalg.norm(table.predict() - ystar, axis=1)
            d[tried] = np.inf
            k = int(np.argmin(d))
        else:  # oracle
            d = np.linalg.norm(response_map(medium, LEVELS) - ystar, axis=1)
            d[tried] = np.inf
            k = int(np.argmin(d))
        y = medium.step(LEVELS[k])
        table.update(k, y, t); t += 1
        tried[k] = True
        if np.linalg.norm(y - ystar) < TOL:
            return step, True, t
    return K, False, t


def run(kind, policy, seed):
    rng = np.random.default_rng(1000 + seed)
    medium, table, t = make(kind, seed), Table(), 0
    steps, hits = [], []
    c0 = getattr(medium, "centers", np.zeros(1)).copy()
    for e in range(EPISODES):
        target_k = int(rng.integers(K))
        ystar = medium.clone().step(LEVELS[target_k])
        n, ok, t = episode(medium, policy, table, ystar, rng, t)
        steps.append(n); hits.append(ok)
    c1 = getattr(medium, "centers", np.zeros(1))
    drift = float(np.linalg.norm(c1 - c0) / (np.linalg.norm(c0) + 1e-12))
    return np.array(steps), np.array(hits), drift


if __name__ == "__main__":
    out = {"tol": TOL, "episodes": EPISODES, "seeds": SEEDS, "K": K, "result": {}}
    for kind in MEDIA:
        out["result"][kind] = {}
        for pol in POLICIES:
            S, H, D = [], [], []
            for s in range(SEEDS):
                st, hi, dr = run(kind, pol, s)
                S.append(st); H.append(hi); D.append(dr)
            S, H = np.array(S), np.array(H)
            first, last = S[:, :20], S[:, -100:]
            per_seed_last = last.mean(1)
            r = {
                "pings_first20": round(float(first.mean()), 2),
                "pings_last100": round(float(last.mean()), 2),
                "pings_last100_se": round(float(per_seed_last.std(ddof=1) / np.sqrt(SEEDS)), 2),
                "success_first20": round(float(H[:, :20].mean()), 3),
                "success_last100": round(float(H[:, -100:].mean()), 3),
                "drift": round(float(np.mean(D)), 3),
                "curve_by_50": [round(float(S[:, i:i + 50].mean()), 2) for i in range(0, EPISODES, 50)],
            }
            out["result"][kind][pol] = r
            print(f"{kind:12s} {pol:7s} pings first20 {r['pings_first20']:5.2f} last100 "
                  f"{r['pings_last100']:5.2f}±{r['pings_last100_se']:.2f}  success "
                  f"{r['success_first20']:.2f}->{r['success_last100']:.2f}  drift {r['drift']:.3f}  "
                  f"curve {r['curve_by_50']}", flush=True)
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/GATE4.json", "w"), indent=1)
