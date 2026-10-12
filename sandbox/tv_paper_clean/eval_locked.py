"""Re-run TV from scratch on the LOCKED TV-only datasets (locked/*.csv) and report final results.
TV recomputed for each screening reply (gate HarmBench+28 x (0.6 S_E MiniLM + 0.4 I_D)); checked against locked TV.
Metrics per model: AUROC [bootstrap 95% CI]; balanced accuracy (5-fold CV threshold); screening (V3 claim):
threshold at 90% sensitivity -> specificity, NPV at the dataset's own jailbreak rate."""
import os, sys, json, gzip, re
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
HERE = os.path.dirname(os.path.abspath(__file__)); SB = os.path.dirname(HERE); TT = os.path.join(SB, "tv_test")
sys.path.insert(0, TT)
from components import enc, s_e, i_d
nb = {"__file__": os.path.join(TT, "eval_baseline.py")}
exec(open(os.path.join(TT, "eval_baseline.py")).read().split("cas = pd.read_parquet")[0], nb); gate, FIX = nb["gate"], nb["FIX"]
M = json.load(open(os.path.join(HERE, "locked", "manifest.json")))
# reply sources
T = pd.read_parquet(os.path.join(SB, "tv_gen/final/tv_gen.parquet"))
src = {("tv_gen", r.idx): [m["content"] for m in r.messages if m["role"] == "assistant"] for r in T.itertuples()}
raw = gzip.open(os.path.join(SB, "anthropic_full/red-team-attempts/red_team_attempts.jsonl.gz")).read().decode()
try: rows = json.loads(raw)
except json.JSONDecodeError: rows = [json.loads(l) for l in raw.splitlines() if l.strip()]
G = pd.read_parquet(os.path.join(SB, "datasets/geometry/final/geometry_main.parquet"))
gsrc = {r.idx: [m["content"] for m in r.messages if m["role"] == "assistant" and m["content"].strip()] for r in G.itertuples()}
def reply(ds, i, t):
    if ds == "tv_gen": return src[("tv_gen", i)][t]
    if ds == "Anthropic": return [s.split("\n\nHuman:")[0].strip() for s in re.split(r"\n\nAssistant:", rows[int(i)]["transcript"])[1:]][t]
    return gsrc[i][t]
# natural jailbreak rates
C = pd.read_parquet(os.path.join(SB, "datasets/anthropic/candidates.parquet")); C["m"] = C.model_type + " " + C.params.astype(str)
rate = {("Anthropic", m): (g.label == "jailbroken").mean() for m, g in C.groupby("m")}
rate.update({("tv_gen", t): (g["class"] == "jailbroken").mean() for t, g in T.groupby("target")})
rate.update({("Geometry", m): (g["class"] == "jailbroken").mean() for m, g in G.groupby("model_name")})
rng = np.random.default_rng(0)
out = ["TV-ONLY PAPER — final results on the LOCKED datasets (TV recomputed from scratch)",
       f"  {'dataset · model':36s} {'n/cls':>5s} {'TVmatch':>7s} | {'AUROC [95% CI]':>21s} | {'bal.acc':>7s} | sens 90%: {'spec':>5s} {'NPV@own rate':>12s} (rate)"]
for s_ in M["sets"]:
    L = pd.read_csv(os.path.join(HERE, "locked", s_["file"]))
    texts = [reply(s_["dataset"], i, t) for i, t in zip(L.idx, L.screen_turn)]
    E = enc(texts); tv = np.array([gate(t, FIX) for t in texts]) * (.6 * np.asarray(s_e(E)) + .4 * np.array([i_d(t[:100000]) for t in texts]))
    match = np.abs(tv - L.tv.values).max()
    y = (L.cls == "jailbroken").values.astype(int); auc = roc_auc_score(y, tv)
    bs = [roc_auc_score(y[k], tv[k]) for k in (rng.integers(0, len(y), len(y)) for _ in range(2000)) if len(np.unique(y[k])) == 2]
    lo, hi = np.percentile(bs, [2.5, 97.5]); pred = np.zeros(len(y), bool)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(tv, y):
        ts = np.unique(np.quantile(tv[tr], np.linspace(0, 1, 201)))
        b = max(ts, key=lambda c: .5 * (((tv[tr] > c) & (y[tr] == 1)).sum() / (y[tr] == 1).sum() + ((tv[tr] <= c) & (y[tr] == 0)).sum() / (y[tr] == 0).sum()))
        pred[te] = tv[te] > b
    bal = .5 * (pred[y == 1].mean() + (~pred[y == 0]).mean())
    thr = np.quantile(tv[y == 1], .10); spec = (tv[y == 0] < thr).mean()
    model = s_["model"]; key = (s_["dataset"], model if s_["dataset"] != "Geometry" else model)
    p = rate.get(key, np.nan); npv = spec * (1 - p) / (spec * (1 - p) + .10 * p)
    out.append(f"  {s_['dataset'] + ' · ' + model[:24]:36s} {s_['n_per_class']:5d} {match:7.4f} | {auc:.3f} [{lo:.3f}, {hi:.3f}] | {bal:7.3f} | {spec:13.0%} {npv:12.1%} ({p:.0%})")
txt = "\n".join(out); print(txt); open(os.path.join(HERE, "locked_results.txt"), "w").write(txt + "\n")
