"""
Gate 4B - why does BLIND pinging get faster over episodes in the plastic medium?
Blind search learns nothing, so any speed-up is the MEDIUM changing.
Measure, before and after 300 goal episodes (rested maps, fatigue cleared):
  redundancy  = mean number of pings whose response lands within TOL of each
                ping's response (how many ways there are to hit a target)
  distinct    = number of response clusters at TOL (greedy cover)
  spread      = summed per-tap std across pings
"""
import os, sys  # run from anywhere: repo root on path, outputs land in pinger/
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(_HERE), _HERE]
os.chdir(_HERE)
import json, numpy as np
import gate4_ping_what_it_wants as g4
from brah.media import response_map
from brah.pinger import LEVELS, K, Table

def stats(m):
    R = response_map(m, LEVELS, rested=True)
    D = np.linalg.norm(R[:, None] - R[None], axis=2)
    left, n = set(range(K)), 0
    while left:
        i = min(left); left -= set(np.flatnonzero(D[i] < g4.TOL)); n += 1
    return dict(redundancy=float((D < g4.TOL).sum(1).mean()), distinct=n, spread=float(R.std(0).sum()))

out = {}
for kind in ["static-net", "habituating", "plastic"]:
    for pol in ["blind", "memory"]:
        before, after = [], []
        for s in range(20):
            rng = np.random.default_rng(1000 + s)
            m, tab, t = g4.make(kind, s), Table(), 0
            before.append(stats(m))
            for e in range(g4.EPISODES):
                ystar = m.clone().step(LEVELS[int(rng.integers(K))])
                _, _, t = g4.episode(m, pol, tab, ystar, rng, t)
            after.append(stats(m))
        agg = {k: [round(float(np.mean([b[k] for b in before])), 3),
                   round(float(np.mean([a[k] for a in after])), 3)] for k in before[0]}
        out[f"{kind}/{pol}"] = agg
        print(kind, pol, agg, flush=True)
json.dump(out, open("results/GATE4B_why_easier.json", "w"), indent=1)
