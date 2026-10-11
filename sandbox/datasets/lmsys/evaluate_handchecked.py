"""LMSYS, hand-checked safety refusals (rules fixed before reading; see handcheck_dump.py).
Models: gpt-4, gpt-3.5-turbo, claude-1, llama-2-13b-chat (published safety training).
Decisions kept: every WildGuard compliance + every refusal read and marked S (true safety refusal) in
handcheck/verdicts.csv. Refusals marked N, and refusals not read, are dropped. TV, PR, habit, splits: unchanged
from evaluate.py (habit = previous reply complied per WildGuard). Output handchecked_results.txt."""
import json, os
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = ["gpt-4", "gpt-3.5-turbo", "claude-1", "llama-2-13b-chat"]
S = pd.read_parquet(os.path.join(HERE, "final", "lmsys_sample.parquet")); S = S[S.model.isin(MODELS)]
P = [json.loads(l) for l in open(os.path.join(HERE, "wg_ds", "pairs.jsonl"))]
W = {int(x["row"]): x["refusal"] for x in map(json.loads, open(os.path.join(HERE, "out_wg", "wg_defcon_rest.jsonl")))}
R = pd.read_parquet(os.path.join(HERE, "final", "lmsys_reply_scores.parquet")).set_index("text")
Q = pd.read_parquet(os.path.join(HERE, "out_pr", "pr_lmsys.parquet")).set_index("text").pr
V = pd.read_csv(os.path.join(HERE, "handcheck", "verdicts.csv")).set_index("k").verdict
CV = pd.read_csv(os.path.join(HERE, "handcheck", "cverdicts.csv")).set_index("k").verdict
MODE = os.environ.get("COMPLY", "all")
FEATS = ["t_gate", "t_s_e", "t_i_d", "t_hedge", "t_len", "pr_msg", "pr_runmax", "pr_prev", "pr_delta"]

# pair index k for every row of the full sample (same walk as prep.py)
Sall = pd.read_parquet(os.path.join(HERE, "final", "lmsys_sample.parquet")); kstart, k = {}, 0
for r in Sall.itertuples():
    kstart[r.conversation_id] = k
    last = None
    for m in r.messages:
        if m["role"] == "user": last = m["content"]
        elif m["role"] == "assistant" and last is not None and m["content"].strip(): k += 1; last = None
rows = []
for r in S.itertuples():
    k, last, pairs = kstart[r.conversation_id], None, []
    for m in r.messages:
        if m["role"] == "user": last = m["content"]
        elif m["role"] == "assistant" and last is not None and m["content"].strip():
            assert (P[k]["prompt"], P[k]["response"]) == (last, m["content"]); pairs.append((k, last, m["content"], W.get(k))); k += 1; last = None
    q = [float(Q[u]) for _, u, _, _ in pairs]
    for i, (kk, u, a, w) in enumerate(pairs):
        if w == "no": keep = CV.get(kk) == "C" if MODE == "read" else CV.get(kk) != "X"
        elif w == "yes": keep = V.get(kk) == "S"
        else: keep = False
        if not keep: continue
        p = R.loc[pairs[i - 1][2]] if i else None
        rows.append({"model": r.model, "conv": r.conversation_id, "complied": int(w == "no"), "first": i == 0,
                     "habit": float(pairs[i - 1][3] == "no") if i else 0.,
                     "t_gate": p.gate if i else 0., "t_s_e": p.s_e if i else 0., "t_i_d": p.i_d if i else 0., "t_hedge": p.hedge if i else 0., "t_len": p.loglen if i else 0.,
                     "pr_msg": q[i], "pr_runmax": max(q[:i + 1]), "pr_prev": q[i - 1] if i else 0., "pr_delta": q[i] - (q[i - 1] if i else 0.)})
D = pd.DataFrame(rows); D["TV"] = D.t_gate * (.6 * D.t_s_e + .4 * D.t_i_d); D["PR"] = D.pr_msg

def bal(p, y): return .5 * (p[y].mean() + (~p[~y]).mean())
def find_c(m, y):
    best, c = -1, 0.0
    for t in np.unique(np.quantile(m, np.linspace(0, 1, 301))):
        b = bal(m > t, y)
        if b > best: best, c = b, t
    return c
folds = lambda g, r: GroupKFold(5, shuffle=True, random_state=r).split(g, groups=g.conv)
def rule(g, m, r):
    y = g.complied.values.astype(bool); m = np.asarray(m, float); p = np.zeros(len(g), bool)
    for tr, te in folds(g, r): p[te] = m[te] > find_c(m[tr], y[tr])
    return bal(p, y)
def logit(g, cols, r, thr=True):
    y = g.complied.values.astype(bool); pr = np.zeros(len(g)); pd_ = np.zeros(len(g), bool)
    for tr, te in folds(g, r):
        mdl = LogisticRegression(max_iter=3000, class_weight="balanced").fit(g.iloc[tr][cols], y[tr]); pr[te] = mdl.predict_proba(g.iloc[te][cols])[:, 1]
        if thr: pd_[te] = pr[te] >= find_c(mdl.predict_proba(g.iloc[tr][cols])[:, 1], y[tr])
    return (bal(pd_, y) if thr else None), roc_auc_score(y, pr)
q = lambda a: f"{np.mean(a):.3f} [{np.min(a):.3f}-{np.max(a):.3f}]"

out = [f"LMSYS-Chat-1M — HAND-CHECKED safety refusals + compliances (COMPLY={MODE}); 20 repeated splits grouped by conversation",
       "balanced accuracy mean [min-max]; AUROC from one split (r=0)", "",
       f"  {'model':17s} {'n':>5s} {'comply':>6s} {'safety ref.':>11s} {'later ref.':>10s} | {'rule all':>19s} {'logistic all':>19s} | {'rule later':>19s} {'habit-PR later':>19s} | AUROC later: PR / TV / both"]
for m in MODELS:
    g = D[D.model == m].reset_index(drop=True); nf = g[~g["first"]].reset_index(drop=True)
    nref = int((1 - nf.complied).sum())
    ra = [rule(g, g.TV - g.PR, r) for r in range(20)]; la = [logit(g, FEATS, r)[0] for r in range(20)]
    rn = [rule(nf, nf.TV - nf.PR, r) for r in range(20)]; hn = [rule(nf, nf.habit - nf.PR, r) for r in range(20)]
    a = [logit(nf, c, 0, False)[1] for c in (["PR"], ["TV"], ["TV", "PR"])]
    out.append(f"  {m:17s} {len(g):5d} {int(g.complied.sum()):6d} {int((1 - g.complied).sum()):11d} {nref:10d} | {q(ra):>19s} {q(la):>19s} | {q(rn):>19s} {q(hn):>19s} | {a[0]:.3f} / {a[1]:.3f} / {a[2]:.3f}")
txt = "\n".join(out); print(txt); open(os.path.join(HERE, f"handchecked_results_{MODE}.txt"), "w").write(txt + "\n")
D.to_parquet(os.path.join(HERE, "final", f"lmsys_handchecked_frame_{MODE}.parquet"), index=False)
