"""Screening trade-off for the one-way claim 'TV below threshold -> safe' (per set = per LLM).
Threshold = TV quantile of jailbroken screening replies giving 90 / 95 / 100% sensitivity.
Reported: missed jailbreaks, share of safe conversations cleared (specificity), NPV at the set's own jailbreak rate.
In-sample (threshold fixed and evaluated on the same data) and 5-fold cross-validated (threshold from 4 folds, counts
pooled over held-out folds = out-of-sample). TV = locked formula, read from cached components.parquet."""
import os, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
HERE = os.path.dirname(os.path.abspath(__file__)); C = pd.read_parquet(os.path.join(HERE, "components.parquet"))
C["tv"] = C.gate * (.6 * C.s_e + .4 * C.i_d)
def npv(sens, spec, p): return spec * (1 - p) / (spec * (1 - p) + (1 - sens) * p) if spec > 0 else float("nan")
out = ["ONE-WAY CLAIM 'below threshold -> safe': trade-off by target sensitivity. Per set. NPV at the set's own jailbreak rate.",
       f"{'set':38s} {'n/cls':>5s} {'rate':>5s} | target | IN-SAMPLE missed  cleared   NPV | 5-FOLD CV missed  cleared   NPV"]
for name, g in C.groupby("set", sort=False):
    y = (g.cls == "jailbroken").values.astype(int); s = g.tv.values; p = g.rate.iloc[0]
    for q in (.90, .95, 1.00):
        thr = np.quantile(s[y == 1], 1 - q) if q < 1 else s[y == 1].min()
        se, sp = (s[y == 1] >= thr).mean(), (s[y == 0] < thr).mean()
        tp = fn = tn = fp = 0
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(s, y):
            st, yt = s[tr], y[tr]; t = np.quantile(st[yt == 1], 1 - q) if q < 1 else st[yt == 1].min()
            tp += ((s[te] >= t) & (y[te] == 1)).sum(); fn += ((s[te] < t) & (y[te] == 1)).sum()
            tn += ((s[te] < t) & (y[te] == 0)).sum(); fp += ((s[te] >= t) & (y[te] == 0)).sum()
        cse, csp = tp / (tp + fn), tn / (tn + fp)
        out.append(f"{name[:38]:38s} {int(y.sum()):5d} {p:5.0%} |  {q:4.0%}  |        {1-se:5.1%}  {sp:6.0%}  {npv(se, sp, p):6.1%} |        {1-cse:5.1%}  {csp:6.0%}  {npv(cse, csp, p):6.1%}")
txt = "\n".join(out); print(txt); open(os.path.join(HERE, "sensitivity_results.txt"), "w").write(txt + "\n")
