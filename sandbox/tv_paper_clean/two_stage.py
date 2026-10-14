"""Two-stage theory on the LOCKED TV-only sets (per LLM, nothing pooled).
Stage 1: TV of the screening reply < threshold (90% sensitivity, as in eval_locked.py) -> no jailbreak.
Stage 2: above the threshold, jailbreak iff TV - PR > c (Mayer Prop 5), PR = perceived risk of the NEXT user request
(JB: the harmful turn; safe: the final turn). Requests: pr_requests.py -> locked_requests.parquet. PR: Kaggle kernel
tv-pr-locked (kaggle_pr/kernel/pr_locked.py, frozen paper method) -> kaggle_pr/out/locked_pr.parquet (+ saved model).
Leak check: requests with MiniLM cos >= .9 to a WildGuardMix train prompt are counted.
Reports, among above-threshold items: AUROC of TV, of -PR, of TV-PR; CV balanced accuracy of each rule; and the full
two-stage rule vs TV-only over the whole set. Writes two_stage_results.txt."""
import os, json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
HERE = os.path.dirname(os.path.abspath(__file__))
P = pd.read_parquet(os.path.join(HERE, "kaggle_pr", "out", "locked_pr.parquet"))
info = json.load(open(os.path.join(HERE, "kaggle_pr", "out", "pr_info.json")))
M = json.load(open(os.path.join(HERE, "locked", "manifest.json")))

def cv_bal(s, y):
    pred = np.zeros(len(y), bool); k = min(5, int(min(y.sum(), (1 - y).sum())))
    if k < 2: return float("nan")
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=0).split(s, y):
        ts = np.unique(np.quantile(s[tr], np.linspace(0, 1, 201)))
        b = max(ts, key=lambda c: .5 * (((s[tr] > c) & (y[tr] == 1)).sum() / (y[tr] == 1).sum() + ((s[tr] <= c) & (y[tr] == 0)).sum() / (y[tr] == 0).sum()))
        pred[te] = s[te] > b
    return .5 * (pred[y == 1].mean() + (~pred[y == 0]).mean())
auc = lambda y, s: roc_auc_score(y, s) if len(set(y)) == 2 else float("nan")

out = ["TWO-STAGE THEORY on the LOCKED TV-only sets. Stage 1: TV < thr (90% sens) -> safe. Stage 2 (above thr): jailbreak iff TV - PR > c.",
       f"PR = perceived risk of the next user request (frozen WildGuardMix method; WildGuardMix test AUROC {info['wgm_test_auroc']:.4f}). Per LLM."]
for s_ in M["sets"]:
    L = P[P.file == s_["file"]].reset_index(drop=True)
    n_all = len(pd.read_csv(os.path.join(HERE, "locked", s_["file"])))
    y = (L.cls == "jailbroken").values.astype(int); tv = L.tv.values; pr = L.pr.values
    thr = np.quantile(tv[y == 1], .10); up = tv >= thr; yu, tu, pu = y[up], tv[up], pr[up]
    two = np.where(up, tv - pr, -9.0)      # stage-1 rejects get the lowest score
    out += [f"\n== {s_['dataset']} · {s_['model']}  (n {int(y.sum())}/{int((1-y).sum())}; no next request: {n_all - len(L)}; PR leak cos>=.9: {int((L.leak_cos >= .9).sum())})",
            f"  PR of next request: JB mean {pr[y==1].mean():.3f} | safe mean {pr[y==0].mean():.3f} | AUROC of -PR on whole set {auc(y, -pr):.3f}",
            f"  above threshold {thr:.3f}: JB {int(yu.sum())}, safe {int((1-yu).sum())}",
            f"    AUROC   TV {auc(yu, tu):.3f} | -PR {auc(yu, -pu):.3f} | TV-PR {auc(yu, tu - pu):.3f}",
            f"    CV bal  TV {cv_bal(tu, yu):.3f} | -PR {cv_bal(-pu, yu):.3f} | TV-PR {cv_bal(tu - pu, yu):.3f}",
            f"  whole set: AUROC TV-only {auc(y, tv):.3f} vs two-stage {auc(y, two):.3f} | CV bal TV-only {cv_bal(tv, y):.3f} vs two-stage {cv_bal(two, y):.3f}"]
txt = "\n".join(out); print(txt); open(os.path.join(HERE, "two_stage_results.txt"), "w").write(txt + "\n")
