"""MAIN per-LLM results with the NEW PR (all-mpnet-base-v2 + logistic regression, WGM-test AUROC .906; scores from
Kaggle sudhanp2004/tv-pr-alternatives column mpnet_lr) and hedge = Do-Not-Answer score only (no phrase list).
Same rows, WildGuard labels and evaluation as eval_rule_vs_model.run (raw rule, rule + per-LLM threshold, 9-feature
logistic; 5-fold GroupKFold by goal). PR features (pr_msg, pr_runmax, pr_prev, pr_delta) recomputed per conversation.
Old = current published main numbers (old PR, old hedge) from the same frames, for comparison."""
import os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "eval_tv_hedge_threshold.py")).read()
ns = {"__file__": os.path.join(HERE, "eval_tv_hedge_threshold.py")}
exec(src.split('out = ["SIMPLE RULE')[0], ns); frames = ns["frames"]
rv = {"__file__": os.path.join(HERE, "eval_rule_vs_model.py")}
exec(open(os.path.join(HERE, "eval_rule_vs_model.py")).read().split("rows = []")[0], rv); run = rv["run"]
QN = pd.read_parquet(os.path.join(HERE, "..", "tv_pr_kaggle", "out", "pr_requests.parquet")).set_index("text").mpnet_lr
HD = {}
for n in ["CASCADE", "tv_gen", "DEFCON", "MTJ"]:
    S = pd.read_parquet(os.path.join(HERE, f"hedge_dna_{n}.parquet")); HD.update(dict(zip(S.text, S.h)))

def pr_feats(reqs):
    q = [float(QN[u]) for u in reqs]
    return [{"pr_msg": q[i], "pr_runmax": max(q[:i + 1]), "pr_prev": q[i - 1] if i else 0., "pr_delta": q[i] - (q[i - 1] if i else 0.)} for i in range(len(q))]

new = {}
for ds, f, pm in [("CASCADE", "../datasets/cascade/final/cascade.parquet", "pm_cascade_wg.parquet"), ("tv_gen", "../tv_gen/final/tv_gen.parquet", "pm_tvgen_wg.parquet")]:
    P = pd.read_parquet(os.path.join(HERE, pm)); rows = []
    for r in pd.read_parquet(os.path.join(HERE, f)).itertuples():
        A = [m["content"] for m in r.messages if m["role"] == "assistant"]; U = [m["content"] for m in r.messages if m["role"] == "user"]
        keep = [(u, a) for u, a in zip(U, A) if a.strip()]
        for i, pf in enumerate(pr_feats([u for u, _ in keep])): rows.append({**pf, "t_hedge": HD[keep[i - 1][1]] if i else 0.})
    X = pd.DataFrame(rows); assert len(X) == len(P)
    for c in X.columns: P[c] = X[c].values
    new.update({(ds, tg): g.reset_index(drop=True) for tg, g in P.groupby("target")})

s2 = open(os.path.join(HERE, "eval_bal_defcon_mtj.py")).read(); nd = {"__file__": os.path.join(HERE, "eval_bal_defcon_mtj.py")}
exec(s2.split("out = [")[0], nd); body = s2.split("# ---- DEF CON")[1]
for ds, sec, lab in [("DEF CON", "# ---- DEF CON" + body.split("for tg, g in frame(convs, Ld, R, Q)")[0], "Ld"),
                     ("MTJ", "# ---- MultiturnJailbreak" + body.split("# ---- MultiturnJailbreak")[1].split("for tg, g in frame(convs, Lm, R, Q)")[0], "Lm")]:
    exec(sec, nd); R = nd["R"].copy(); R["hedge"] = [HD[t] for t in R.index]
    for tg, g in nd["frame"](nd["convs"], nd[lab], R, QN).groupby("target"): new[(ds, tg)] = g.reset_index(drop=True)

f3 = lambda t: f"{t[0]:.3f} ({t[1]:.2f}/{t[2]:.2f})"
out = ["MAIN RESULTS — NEW PR (mpnet + logistic regression) and Do-Not-Answer-only hedge vs OLD (MiniLM PR, hedge with phrase list)",
       "held-out bal.acc (compliances caught / refusals caught); 5-fold by goal; WildGuard labels", "",
       f"  {'dataset':8s} {'LLM':26s} {'n':>6s} | {'raw old':>7s} {'raw new':>7s} | {'thr old':>7s} {'thr new':>18s} | {'logit old':>9s} {'logit new':>18s} | AUROC rule old/new"]
S = []
for ds, tg, g in frames:
    n = new[(ds, tg)]; assert len(n) == len(g) and (n.complied.values == g.complied.values).all(), (ds, tg)
    a, b = run(g), run(n); S.append((a, b))
    out.append(f"  {ds:8s} {tg[:26]:26s} {len(g):6d} | {a['raw'][0]:7.3f} {b['raw'][0]:7.3f} | {a['thr'][0]:7.3f} {f3(b['thr']):>18s} | "
               f"{a['model'][0]:9.3f} {f3(b['model']):>18s} | {a['auc_margin']:.3f}/{b['auc_margin']:.3f}")
med = lambda k, j: np.median([s[j][k][0] for s in S])
out.append(f"  {'median':42s} | {med('raw', 0):7.3f} {med('raw', 1):7.3f} | {med('thr', 0):7.3f} {med('thr', 1):18.3f} | {med('model', 0):9.3f} {med('model', 1):18.3f}")
txt = "\n".join(out); print(txt); open(os.path.join(HERE, "main_results_newpr.txt"), "w").write(txt + "\n")
