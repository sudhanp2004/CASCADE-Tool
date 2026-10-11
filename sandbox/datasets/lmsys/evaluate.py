"""Trust-risk theory on LMSYS-Chat-1M (real users, 20 named models), exactly the paper's setup:
TV = gate*(.6 S_E + .4 I_D) of the previous reply (0 at a conversation's first decision); PR = mpnet+LR on the current
request (Kaggle tv-pr-lmsys); labels = WildGuard 7B (complied = 'Response refusal: no'; Kaggle tv-wildguard-lmsys).
Per model, 20 repeated GroupKFold(5, shuffle) splits grouped by CONVERSATION (no goals in LMSYS):
  all decisions: rule TV - PR > c | 9-feature logistic
  later decisions: rule | habit - PR | AUROC PR alone / TV alone / TV+PR
Models need >= 20 refusals and >= 20 compliances among later decisions to be tested."""
import json, os
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
S = pd.read_parquet(os.path.join(HERE, "final", "lmsys_sample.parquet"))
P = [json.loads(l) for l in open(os.path.join(HERE, "wg_ds", "pairs.jsonl"))]
W = {int(x["row"]): x["refusal"] for x in map(json.loads, open(os.path.join(HERE, "out_wg", "wg_defcon_rest.jsonl")))}
R = pd.read_parquet(os.path.join(HERE, "final", "lmsys_reply_scores.parquet")).set_index("text")
Q = pd.read_parquet(os.path.join(HERE, "out_pr", "pr_lmsys.parquet")).set_index("text").pr
auc_wgm = json.load(open(os.path.join(HERE, "out_pr", "pr_lmsys_auc.json")))["wgm_test_auroc"]
FEATS = ["t_gate", "t_s_e", "t_i_d", "t_hedge", "t_len", "pr_msg", "pr_runmax", "pr_prev", "pr_delta"]

rows, k = [], 0
for r in S.itertuples():
    last, pairs = None, []
    for m in r.messages:
        if m["role"] == "user": last = m["content"]
        elif m["role"] == "assistant" and last is not None and m["content"].strip():
            assert (P[k]["prompt"], P[k]["response"]) == (last, m["content"]); pairs.append((last, m["content"], W.get(k))); k += 1; last = None
    q = [float(Q[u]) for u, _, _ in pairs]
    for i, (u, a, w) in enumerate(pairs):
        if w not in ("yes", "no"): continue
        p = R.loc[pairs[i - 1][1]] if i else None
        rows.append({"model": r.model, "conv": r.conversation_id, "flagged": r.flagged, "complied": int(w == "no"), "first": i == 0,
                     "habit": float(pairs[i - 1][2] == "no") if i else 0.,
                     "t_gate": p.gate if i else 0., "t_s_e": p.s_e if i else 0., "t_i_d": p.i_d if i else 0., "t_hedge": p.hedge if i else 0., "t_len": p.loglen if i else 0.,
                     "pr_msg": q[i], "pr_runmax": max(q[:i + 1]), "pr_prev": q[i - 1] if i else 0., "pr_delta": q[i] - (q[i - 1] if i else 0.)})
D = pd.DataFrame(rows); D["TV"] = D.t_gate * (.6 * D.t_s_e + .4 * D.t_i_d); D["PR"] = D.pr_msg
D.to_parquet(os.path.join(HERE, "final", "lmsys_frame.parquet"), index=False)

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
q = lambda a: f"{np.mean(a):.3f}"
out = [f"LMSYS-Chat-1M — trust-risk rule per model (WildGuard labels; PR mpnet+LR, WildGuardMix test AUROC {auc_wgm:.3f}); 20 repeated splits grouped by conversation",
       f"labelled decisions {len(D)} (WildGuard label missing for {sum(1 for x in W.values() if x not in ('yes','no'))})", "",
       f"  {'model':22s} {'n':>5s} {'later':>5s} {'refus.':>6s} {'comply':>6s} | {'rule all':>8s} {'logit':>6s} | {'rule later':>10s} {'habit-PR':>8s} | AUROC later: PR / TV / both"]
for m, g in sorted(D.groupby("model"), key=lambda kv: -len(kv[1])):
    g = g.reset_index(drop=True); nf = g[~g["first"]].reset_index(drop=True)
    nref = int((1 - nf.complied).sum()); ncom = int(nf.complied.sum())
    if min(nref, ncom) < 20: out.append(f"  {m[:22]:22s} {len(g):5d} {len(nf):5d} {nref:6d} {g.complied.mean():6.0%} | too few later refusals/compliances to test"); continue
    ra = [rule(g, g.TV - g.PR, r) for r in range(20)]; la = [logit(g, FEATS, r)[0] for r in range(20)]
    rn = [rule(nf, nf.TV - nf.PR, r) for r in range(20)]; hn = [rule(nf, nf.habit - nf.PR, r) for r in range(20)]
    a = [logit(nf, c, 0, False)[1] for c in (["PR"], ["TV"], ["TV", "PR"])]
    out.append(f"  {m[:22]:22s} {len(g):5d} {len(nf):5d} {nref:6d} {g.complied.mean():6.0%} | {q(ra):>8s} {q(la):>6s} | {q(rn):>10s} {q(hn):>8s} | {a[0]:.3f} / {a[1]:.3f} / {a[2]:.3f}")
txt = "\n".join(out); print(txt); open(os.path.join(HERE, "lmsys_results.txt"), "w").write(txt + "\n")
