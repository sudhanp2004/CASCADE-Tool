"""Main per-LLM results with REPEATED random grouped cross-validation (fixes the fold-tie dependence found 2026-10-09):
20 repeats of GroupKFold(5, shuffle=True, random_state=r), groups = goal (DEF CON: challenge). Reported: mean [min, max].
Run only with the base environment (/home/sudhan/miniconda3/bin/python, scikit-learn 1.8.0).
Setup: TV = MiniLM gate*(.6 S_E + .4 I_D) of the previous reply; PR = mpnet + logistic regression; hedge = Do-Not-Answer;
labels = WildGuard 7B. Datasets: tv_gen; DEF CON with the reset fix (POOLED over up to 8 unidentified LLMs); MTJ.
CASCADE excluded (user decision). Per LLM:
  all decisions:   rule TV - PR > c (c per fold on training goals) | 9-feature logistic
  later decisions: rule | habit - PR rule | logistic AUROC with PR alone, TV alone, TV + PR"""
import os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "eval_main_newpr.py")).read(); ns = {"__file__": os.path.join(HERE, "eval_main_newpr.py")}
exec(src.split("f3 = lambda")[0], ns); new = ns["new"]
dc = {"__file__": os.path.join(HERE, "eval_defcon_reset_fix.py")}
exec(open(os.path.join(HERE, "eval_defcon_reset_fix.py")).read().split("old = new[")[0], dc); defcon = dc["fixed"]
FEATS = ["t_gate", "t_s_e", "t_i_d", "t_hedge", "t_len", "pr_msg", "pr_runmax", "pr_prev", "pr_delta"]
REPEATS = 20

# habit = WildGuard label of the previous reply (0 at a first decision)
def add_habit_keyed(g, prev):
    lab = dict(zip(g.key, g.complied)); g = g.copy()
    g["first"] = g.t_len == 0
    g["habit"] = [0. if f else lab.get(prev(k), 0.) for k, f in zip(g.key, g["first"])]
    return g
F = {}
P = pd.read_parquet(os.path.join(HERE, "pm_tvgen_wg.parquet")); conv = []
for r in pd.read_parquet(os.path.join(HERE, "..", "tv_gen", "final", "tv_gen.parquet")).itertuples():
    A = [m["content"] for m in r.messages if m["role"] == "assistant"]; U = [m["content"] for m in r.messages if m["role"] == "user"]
    conv += [r.idx] * sum(1 for u, a in zip(U, A) if a.strip())
P["conv"] = conv; P["habit"] = P.groupby("conv").complied.shift(1).fillna(0.)
for tg, idx in P.groupby("target").groups.items():
    g = new[("tv_gen", tg)].copy(); g["habit"] = P.loc[idx, "habit"].values; g["first"] = g.t_len == 0; F[("tv_gen", tg)] = g
F[("DEF CON", "pooled, up to 8 LLMs")] = add_habit_keyed(defcon, lambda k: (k[0], k[1], k[2] - 1))
for (ds, tg), g in new.items():
    if ds == "MTJ": F[("MTJ", tg)] = add_habit_keyed(g, lambda k: (k[0], k[1][:-1]))

def bal(p, y): return .5 * (p[y].mean() + (~p[~y]).mean())
def find_c(m, y):
    best, c = -1, 0.0
    for t in np.unique(np.quantile(m, np.linspace(0, 1, 301))):
        b = bal(m > t, y)
        if b > best: best, c = b, t
    return c
def folds(g, r): return GroupKFold(5, shuffle=True, random_state=r).split(g, groups=g.goal)
def rule(g, m, r):
    y = g.complied.values.astype(bool); m = np.asarray(m, float); p = np.zeros(len(g), bool)
    for tr, te in folds(g, r): p[te] = m[te] > find_c(m[tr], y[tr])
    return bal(p, y)
def logit(g, cols, r, thr=True):
    y = g.complied.values.astype(bool); pr = np.zeros(len(g)); pd_ = np.zeros(len(g), bool)
    for tr, te in folds(g, r):
        mdl = LogisticRegression(max_iter=3000, class_weight="balanced").fit(g.iloc[tr][cols], y[tr])
        pr[te] = mdl.predict_proba(g.iloc[te][cols])[:, 1]
        if thr: pd_[te] = pr[te] >= find_c(mdl.predict_proba(g.iloc[tr][cols])[:, 1], y[tr])
    return (bal(pd_, y) if thr else None), roc_auc_score(y, pr)

rows = []
for (ds, tg), g in F.items():
    g = g.reset_index(drop=True); g["TV"] = g.t_gate * (.6 * g.t_s_e + .4 * g.t_i_d); g["PR"] = g.pr_msg
    nf = g[~g["first"]].reset_index(drop=True); y = g.complied.values
    ok_nf = min(nf.complied.sum(), (1 - nf.complied).sum()) >= 10
    R = {k: [] for k in ["rule", "logit", "rule_nf", "habit_nf", "auc_pr", "auc_tv", "auc_both"]}
    for r in range(REPEATS):
        R["rule"].append(rule(g, g.TV - g.PR, r)); R["logit"].append(logit(g, FEATS, r)[0])
        if ok_nf:
            R["rule_nf"].append(rule(nf, nf.TV - nf.PR, r)); R["habit_nf"].append(rule(nf, nf.habit - nf.PR, r))
            R["auc_pr"].append(logit(nf, ["PR"], r, False)[1]); R["auc_tv"].append(logit(nf, ["TV"], r, False)[1])
            R["auc_both"].append(logit(nf, ["TV", "PR"], r, False)[1])
    rows.append((ds, tg, len(g), y.mean(), roc_auc_score(y, g.TV - g.PR), R)); print(ds, tg, "done", flush=True)

q = lambda a: f"{np.mean(a):.3f} [{np.min(a):.3f}-{np.max(a):.3f}]" if len(a) else "n/a"
out = [f"MAIN RESULTS — {REPEATS} repeated random grouped 5-fold splits; mean [min-max] over repeats (base env, sklearn 1.8.0)",
       "CASCADE excluded; DEF CON pooled over up to 8 unidentified LLMs (reset fix applied)", ""]
for ds, tg, n, cr, auc, R in rows:
    out += [f"== {ds} | {tg}  (n={n}, complied {cr:.0%}, rule AUROC {auc:.3f})",
            f"   all decisions:   rule + threshold {q(R['rule'])} | logistic {q(R['logit'])}",
            f"   later decisions: rule {q(R['rule_nf'])} | habit - PR {q(R['habit_nf'])}",
            f"   later, AUROC:    PR alone {q(R['auc_pr'])} | TV alone {q(R['auc_tv'])} | TV + PR {q(R['auc_both'])}"]
txt = "\n".join(out); print(txt); open(os.path.join(HERE, "main_results_repeated_cv.txt"), "w").write(txt + "\n")
