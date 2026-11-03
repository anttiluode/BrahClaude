"""G1 — identify a hidden edge type by probing it (paired-pulse protocol, generalised)."""
import numpy as np

from common import bayes_accuracy, save
from brah.edges import TYPES, kernel

NAMES = ["FAC", "DEP", "BAND", "AUG"]
P, WIN, SIGMA = 6, 128, 0.3
KAP = np.stack([kernel(TYPES[n], WIN) for n in NAMES])  # (4, WIN), kappa(s=1..WIN)


def templates(times):
    t = np.sort(np.asarray(times, int))
    lag = t[:, None] - t[None, :]                # (P, P)
    mask = lag > 0
    k = np.where(mask[None], KAP[:, np.clip(lag - 1, 0, WIN - 1)], 0.0)  # (4,P,P)
    return np.maximum(0.0, 1.0 + k.sum(-1))      # (4, P)


def score_crn(times, labels, noise):
    tp = templates(times)
    obs = tp[labels] + noise
    d2 = ((obs[:, None, :] - tp[None]) ** 2).sum(-1)
    return (d2.argmin(1) == labels).mean()


def main():
    rng = np.random.default_rng(1)
    fixed = {
        "paired_isi3": [0, 3, 40, 43, 80, 83],
        "regular_isi21": [0, 21, 42, 63, 84, 105],
        "burst_isi1": [0, 1, 2, 3, 4, 5],
    }
    acc = {k: bayes_accuracy(templates(v), SIGMA, 20000, rng) for k, v in fixed.items()}
    rand_accs = []
    for _ in range(200):
        tt = np.sort(rng.choice(WIN, P, replace=False))
        rand_accs.append(bayes_accuracy(templates(tt), SIGMA, 2000, rng))
    acc["random_mean"] = float(np.mean(rand_accs))

    # cross-entropy search over pulse positions (scored on a fixed noise sample)
    labels = rng.integers(0, 4, 4000)
    noise = rng.normal(0, SIGMA, (4000, P))
    logits = np.zeros(WIN)
    best, best_t = -1, None
    for it in range(80):
        cands, scores = [], []
        for _ in range(120):
            gumb = logits + rng.gumbel(size=WIN)
            tt = np.sort(np.argsort(-gumb)[:P])
            cands.append(tt); scores.append(score_crn(tt, labels, noise))
        order = np.argsort(scores)[::-1][:12]
        if scores[order[0]] > best:
            best, best_t = scores[order[0]], cands[order[0]]
        freq = np.zeros(WIN)
        for i in order:
            freq[cands[i]] += 1
        p = 0.7 * np.exp(logits) / np.exp(logits).sum() + 0.3 * freq / freq.sum()
        logits = np.log(p + 1e-9)
    acc["optimised"] = bayes_accuracy(templates(best_t), SIGMA, 20000, np.random.default_rng(99))

    baseline = max(acc["paired_isi3"], acc["regular_isi21"], acc["burst_isi1"], acc["random_mean"])
    res = {
        "accuracy": acc,
        "optimised_train": best_t,
        "optimised_templates": templates(best_t).round(3),
        "best_baseline": baseline,
        "pass": acc["optimised"] >= 0.90 and acc["optimised"] >= baseline + 0.05,
    }
    print(res)
    save("g1", res)


if __name__ == "__main__":
    main()
