"""TV-fit selection + balanced early-warning test (TV-only paper). Selection rule (stated in the paper):
  safe conversations where the model visibly refused at least once (gate<0.5 on some reply) - TV's mechanism applies;
  jailbroken conversations with at least one reply before the first harmful reply;
  equal counts per model (random sample, seed 0).
One screening point per conversation, symmetric: JB -> TV of the reply just before the first harmful reply;
safe -> TV of the reply just before the final turn. Per model: AUROC (bootstrap CI over conversations), and
balanced accuracy / caught / left alone at the threshold maximising balanced accuracy (5-fold CV threshold)."""
import os, sys, json, gzip, re
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
HERE = os.path.dirname(os.path.abspath(__file__)); SB = os.path.dirname(HERE); TT = os.path.join(SB, "tv_test")
sys.path.insert(0, TT)
nb = {"__file__": os.path.join(TT, "eval_baseline.py")}
exec(open(os.path.join(TT, "eval_baseline.py")).read().split("cas = pd.read_parquet")[0], nb); gate, FIX = nb["gate"], nb["FIX"]
rng = np.random.default_rng(0)

def refusal_table(convs):  # {idx: [replies]} -> {(idx,turn): refused}
    return {(i, t): gate(a, FIX) < 0.5 for i, reps in convs.items() for t, a in enumerate(reps)}

# Geometry
G = pd.read_parquet(os.path.join(SB, "datasets/geometry/final/geometry_main.parquet"))
gref = refusal_table({r.idx: [m["content"] for m in r.messages if m["role"] == "assistant" and m["content"].strip()] for r in G.itertuples()})
geo = pd.read_parquet(os.path.join(HERE, "geo_tv.parquet")).rename(columns={"model_name": "model", "harm_turn": "harm"})
geo["ds"] = "Geometry"
# Anthropic
raw = gzip.open(os.path.join(SB, "anthropic_full/red-team-attempts/red_team_attempts.jsonl.gz")).read().decode()
try: rows = json.loads(raw)
except json.JSONDecodeError: rows = [json.loads(l) for l in raw.splitlines() if l.strip()]
an = pd.read_parquet(os.path.join(HERE, "anth_tv.parquet")).rename(columns={"first_harm": "harm"})
an["cls"] = an.cls.replace({"jailbroken": "jailbroken", "safe": "safe"}); an["ds"] = "Anthropic"
aref = refusal_table({i: [s.split("\n\nHuman:")[0].strip() for s in re.split(r"\n\nAssistant:", rows[int(i)]["transcript"])[1:]] for i in an.idx.unique()})

out = ["TV-FIT SELECTION, balanced, symmetric screening point (TV-only early warning)",
       f"  {'dataset':10s} {'model':28s} {'n/class':>7s} | {'AUROC [95% CI]':>21s} | {'bal.acc':>7s} {'caught':>6s} {'left alone':>10s} | mean TV JB / safe"]
for df, ref in ((geo, gref), (an, aref)):
    df = df.copy(); df["refused"] = [ref.get((i, t), False) for i, t in zip(df.idx, df.turn)]
    for m, g in df.groupby("model"):
        last = g.groupby("idx").turn.max()
        sref = g[g.cls == "safe"].groupby("idx").refused.any()
        safe_ids = [i for i in sref[sref].index if last[i] >= 1]
        jb = g[(g.cls == "jailbroken") & (g.harm >= 1)]; jb_ids = list(jb.idx.unique())
        n = min(len(safe_ids), len(jb_ids))
        if n < 20: out.append(f"  {g.ds.iloc[0]:10s} {m[:28]:28s} too few (safe-fit {len(safe_ids)}, JB {len(jb_ids)})"); continue
        si = rng.choice(safe_ids, n, replace=False); ji = rng.choice(jb_ids, n, replace=False)
        hm = dict(zip(jb.idx, jb.harm))
        P = np.array([g[(g.idx == i) & (g.turn == int(hm[i]) - 1)].tv.iloc[0] for i in ji])
        S = np.array([g[(g.idx == i) & (g.turn == last[i] - 1)].tv.iloc[0] for i in si])
        y = np.r_[np.ones(n), np.zeros(n)]; s = np.r_[P, S]; auc = roc_auc_score(y, s)
        bs = [roc_auc_score(y[k], s[k]) for k in (rng.integers(0, 2 * n, 2 * n) for _ in range(1000)) if len(np.unique(y[k])) == 2]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        pred = np.zeros(2 * n, bool)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(s, y):
            ts = np.unique(np.quantile(s[tr], np.linspace(0, 1, 201)))
            best = max(ts, key=lambda c: .5 * (((s[tr] > c) & (y[tr] == 1)).sum() / (y[tr] == 1).sum() + ((s[tr] <= c) & (y[tr] == 0)).sum() / (y[tr] == 0).sum()))
            pred[te] = s[te] > best
        caught = pred[y == 1].mean(); alone = (~pred[y == 0]).mean()
        out.append(f"  {g.ds.iloc[0]:10s} {m[:28]:28s} {n:7d} | {auc:.3f} [{lo:.3f}, {hi:.3f}] | {.5*(caught+alone):7.3f} {caught:6.0%} {alone:10.0%} | {P.mean():.3f} / {S.mean():.3f}")
txt = "\n".join(out); print(txt); open(os.path.join(HERE, "select_and_test_results.txt"), "w").write(txt + "\n")
