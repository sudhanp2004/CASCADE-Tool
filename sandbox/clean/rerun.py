"""Clean rerun: every experiment on decisions that are true to the theory (rules fixed 2026-10-11 before reading).
IN SCOPE = a true compliance, or a safety refusal (declined/deflected/lectured BECAUSE the content is harmful,
unethical or inappropriate). OUT = refusals of false facts, capability/identity answers, clarification requests,
broken output, partial answers withholding the key part.
Keep rule per (dataset, LLM, WildGuard label), fixed before reading:
  in-scope share among READ items >= 95%  -> keep the whole set (read items marked out of scope are still dropped)
  otherwise                               -> keep only read items verified in scope
A compliance verified as a safety refusal is relabelled (complied = 0); habit of the following decision is NOT
recomputed for those few items. DEF CON: only the 7 challenges whose success means harmful output. MTJ: left out
(not yet verified). LMSYS: the four safety-trained models, verdicts from the earlier hand check.
Outputs clean_results.txt."""
import json, os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__)); SB = os.path.dirname(HERE); TT = os.path.join(SB, "tv_test")
src = open(os.path.join(TT, "eval_repeated_cv.py")).read(); ns = {"__file__": os.path.join(TT, "eval_repeated_cv.py")}
exec(src.split("rows = []")[0], ns)
F, bal, find_c, logit_, rule_ = ns["F"], ns["bal"], ns["find_c"], ns["logit"], ns["rule"]
FEATS = ns["FEATS"]; REPEATS = 20
V = pd.read_csv(os.path.join(HERE, "verdicts.csv"))

def keyof(ds, tg, g):
    if ds == "tv_gen": return [json.dumps([tg, i]) for i in range(len(g))]
    if ds == "DEF CON": return [json.dumps([int(k[0]), int(k[1]), int(k[2])]) for k in g.key]
    return [json.dumps([k[0], [int(x) for x in k[1]]]) for k in g.key]

out, kept_note = [], []
clean = {}
for (ds, tg), g in F.items():
    if ds == "MTJ": continue
    g = g.reset_index(drop=True).copy(); g["ckey"] = keyof(ds, tg, g)
    g["TV"] = g.t_gate * (.6 * g.t_s_e + .4 * g.t_i_d); g["PR"] = g.pr_msg
    v = V[(V.ds == ds) & (V.model == tg)].set_index("key").verdict
    g["verdict"] = g.ckey.map(v)
    keep = np.zeros(len(g), bool)
    for lab, name in ((1, "comply"), (0, "refuse")):
        m = g.complied == lab; read = m & g.verdict.notna()
        good = "C" if lab else "S"
        share = (g.verdict[read] == good).mean() if read.sum() else 0.0
        if share >= .95: keep |= m & (g.verdict.isna() | (g.verdict == good))
        else: keep |= read & (g.verdict == good)
        kept_note.append(f"  {ds:8s} {str(tg)[:20]:20s} {name:6s} read {int(read.sum()):5d} of {int(m.sum()):6d} | in scope {share:5.1%} -> "
                         f"{'keep all' if share >= .95 else 'keep verified only'} | kept {int((keep & m).sum()):5d}")
    g.loc[(g.complied == 1) & (g.verdict == "S"), "complied"] = 0      # verified safety refusal mislabelled as compliance
    keep |= (g.complied == 0) & (g.verdict == "S")
    clean[(ds, tg)] = g[keep].reset_index(drop=True)

# LMSYS: four safety-trained models, verdicts from the earlier hand check (refusals read one by one, compliances sampled)
L = os.path.join(SB, "datasets", "lmsys")
ls = open(os.path.join(L, "evaluate_handchecked.py")).read().split("def bal")[0]
lns = {"__file__": os.path.join(L, "evaluate_handchecked.py")}; exec(ls, lns)
for m, g in lns["D"].groupby("model"):
    g = g.reset_index(drop=True).copy(); g["goal"] = g.conv; clean[("LMSYS", m)] = g

def rule(g, m, r):
    y = g.complied.values.astype(bool); m = np.asarray(m, float); p = np.zeros(len(g), bool)
    for tr, te in GroupKFold(5, shuffle=True, random_state=r).split(g, groups=g.goal): p[te] = m[te] > find_c(m[tr], y[tr])
    return bal(p, y)
def logit(g, cols, r, thr=True):
    y = g.complied.values.astype(bool); pr = np.zeros(len(g)); pd_ = np.zeros(len(g), bool)
    for tr, te in GroupKFold(5, shuffle=True, random_state=r).split(g, groups=g.goal):
        mdl = LogisticRegression(max_iter=3000, class_weight="balanced").fit(g.iloc[tr][cols], y[tr]); pr[te] = mdl.predict_proba(g.iloc[te][cols])[:, 1]
        if thr: pd_[te] = pr[te] >= find_c(mdl.predict_proba(g.iloc[tr][cols])[:, 1], y[tr])
    return (bal(pd_, y) if thr else None), roc_auc_score(y, pr)
q = lambda a: f"{np.mean(a):.3f} [{np.min(a):.3f}-{np.max(a):.3f}]"

out += ["CLEAN RERUN — only true compliances and safety refusals (MTJ left out, not yet verified)", "",
        "KEEP RULE APPLIED (read = hand-checked by Claude, request+reply only, random order, TV/PR never shown)", *kept_note, "",
        "MAIN TABLE — 20 repeated grouped 5-fold splits; balanced accuracy mean [min-max]",
        f"  {'dataset':8s} {'LLM':20s} {'n':>5s} {'later':>5s} {'refus.':>6s} {'comply':>6s} | {'rule all':>19s} {'logistic all':>19s} | {'rule later':>19s} {'habit-PR later':>19s} | AUROC later: PR / TV / both"]
for (ds, tg), g in clean.items():
    nf = g[~g["first"]].reset_index(drop=True); nref = int((1 - nf.complied).sum()); ncom = int(nf.complied.sum())
    if min(nref, ncom) < 20:
        out.append(f"  {ds:8s} {str(tg)[:20]:20s} {len(g):5d} {len(nf):5d} {nref:6d} {g.complied.mean():6.0%} | too few later refusals or compliances"); continue
    ra = [rule(g, g.TV - g.PR, r) for r in range(REPEATS)]; la = [logit(g, FEATS, r)[0] for r in range(REPEATS)]
    rn = [rule(nf, nf.TV - nf.PR, r) for r in range(REPEATS)]; hn = [rule(nf, nf.habit - nf.PR, r) for r in range(REPEATS)]
    a = [logit(nf, c, 0, False)[1] for c in (["PR"], ["TV"], ["TV", "PR"])]
    out.append(f"  {ds:8s} {str(tg)[:20]:20s} {len(g):5d} {len(nf):5d} {nref:6d} {g.complied.mean():6.0%} | {q(ra):>19s} {q(la):>19s} | {q(rn):>19s} {q(hn):>19s} | {a[0]:.3f} / {a[1]:.3f} / {a[2]:.3f}")

# two factors, later decisions: paired cluster bootstrap on the AUROC gain, Holm across LLMs
out += ["", "TWO FACTORS (later decisions, held-out logistic AUROC; paired cluster bootstrap over groups, Holm)",
        f"  {'dataset':8s} {'LLM':20s} {'n':>5s} {'PR':>6s} {'TV':>6s} {'both':>6s} | {'TV adds':>8s} {'p':>7s} | {'PR adds':>8s} {'p':>7s}"]
def oof(g, cols):
    y = g.complied.values.astype(bool); pr = np.zeros(len(g))
    for tr, te in GroupKFold(5, shuffle=True, random_state=0).split(g, groups=g.goal):
        pr[te] = LogisticRegression(max_iter=3000, class_weight="balanced").fit(g.iloc[tr][cols], y[tr]).predict_proba(g.iloc[te][cols])[:, 1]
    return pr
rows, pv = [], {}
for (ds, tg), g in clean.items():
    nf = g[~g["first"]].reset_index(drop=True)
    if min(int((1 - nf.complied).sum()), int(nf.complied.sum())) < 20: continue
    y = nf.complied.values.astype(bool); P1, P2, P3 = oof(nf, ["PR"]), oof(nf, ["TV"]), oof(nf, ["TV", "PR"])
    a = [roc_auc_score(y, P1), roc_auc_score(y, P2), roc_auc_score(y, P3)]
    rng = np.random.default_rng(0); gr = nf.goal.values; u = np.unique(gr); idx = {k: np.flatnonzero(gr == k) for k in u}
    d1, d2 = [], []
    for _ in range(2000):
        s = np.concatenate([idx[k] for k in rng.choice(u, len(u))])
        if len(np.unique(y[s])) < 2: continue
        d1.append(roc_auc_score(y[s], P3[s]) - roc_auc_score(y[s], P1[s])); d2.append(roc_auc_score(y[s], P3[s]) - roc_auc_score(y[s], P2[s]))
    p1 = min(1., 2 * min(np.mean(np.array(d1) <= 0), np.mean(np.array(d1) >= 0))); p2 = min(1., 2 * min(np.mean(np.array(d2) <= 0), np.mean(np.array(d2) >= 0)))
    rows.append([ds, tg, len(nf), a, a[2] - a[0], p1, a[2] - a[1], p2])
for col, j in (("tv", 5), ("pr", 7)):
    o = sorted(range(len(rows)), key=lambda i: rows[i][j]); run = 0.
    for r, i in enumerate(o): run = max(run, min(1., (len(o) - r) * rows[i][j])); pv[(i, col)] = run
for i, (ds, tg, n, a, dtv, _, dpr, _) in enumerate(rows):
    out.append(f"  {ds:8s} {str(tg)[:20]:20s} {n:5d} {a[0]:6.3f} {a[1]:6.3f} {a[2]:6.3f} | {dtv:+8.3f} {pv[(i,'tv')]:7.3f} | {dpr:+8.3f} {pv[(i,'pr')]:7.3f}")

txt = "\n".join(out); print(txt); open(os.path.join(HERE, "clean_results.txt"), "w").write(txt + "\n")
for k, g in clean.items(): g.drop(columns=[c for c in ("reply", "request") if c in g], errors="ignore").to_parquet(
    os.path.join(HERE, f"frame_{k[0].replace(' ', '')}_{str(k[1]).replace('/', '_')[:24]}.parquet"), index=False)
