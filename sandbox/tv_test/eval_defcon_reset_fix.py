"""DEF CON 31 with the reset fix: a 'reset' message clears the model's context, so each conversation is split at every
reset into separate segments (TV = 0 and PR history restart in each segment). Everything else as eval_main_newpr.py:
new PR (mpnet + LR), MiniLM TV, Do-Not-Answer hedge, WildGuard labels, groups = challenge.
NOTE: DEF CON pools up to 8 unidentified LLMs (identities scrubbed by the organisers) - population-level check only."""
import json, os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "eval_main_newpr.py")).read(); ns = {"__file__": os.path.join(HERE, "eval_main_newpr.py")}
exec(src.split("f3 = lambda")[0], ns)
nd, QN, HD, run, new = ns["nd"], ns["QN"], ns["HD"], ns["run"], ns["new"]

s2 = open(os.path.join(HERE, "eval_bal_defcon_mtj.py")).read(); body = s2.split("# ---- DEF CON")[1]
exec("# ---- DEF CON" + body.split("for tg, g in frame(convs, Ld, R, Q)")[0], nd)   # Rd, Ld, R (reply scores)
R = nd["R"].copy(); R["hedge"] = [HD[t] for t in R.index]

convs, n_split = [], 0
for ci, r in enumerate(nd["Rd"]):
    seg, last, pairs = 0, "", []
    def flush():
        if pairs: convs.append({"llm": "DEF CON (pooled, up to 8 LLMs)", "goal": r["challenges_name"] or "(none)",
                                "pairs": list(pairs), "key": (lambda ci, sg: (lambda i: (ci, sg, i)))(ci, seg)})
    for m in r["conversation"]:
        role, b = m.get("role", ""), (m.get("body") or "").strip()
        if role.startswith("reset"):
            if pairs: flush(); n_split += 1
            seg, last, pairs = seg + 1, "", []
            continue
        if role.startswith("user"): last = b
        elif role.startswith("assistant") and b and last: pairs.append((last, b)); last = ""
    flush()
fixed = nd["frame"](convs, nd["Ld"], R, QN).reset_index(drop=True)
old = new[("DEF CON", "DEF CON model (anonymized)")]
f = lambda t: f"{t[0]:.3f} ({t[1]:.2f}/{t[2]:.2f})"
out = ["DEF CON 31 — reset fix (conversations split at every reset); POOLED over up to 8 unidentified LLMs",
       f"segments created by resets: {n_split} | decisions: before {len(old)}, after {len(fixed)} | first decisions: before {int((old.t_len == 0).sum())}, after {int((fixed.t_len == 0).sum())}", ""]
for name, g in [("before fix", old), ("after fix", fixed)]:
    a = run(g.reset_index(drop=True)); nf = g[g.t_len > 0].reset_index(drop=True); b = run(nf)
    out += [f"  {name:10s} all decisions: rule raw {f(a['raw'])} | rule + threshold {f(a['thr'])} | logistic {f(a['model'])} | AUROC rule {a['auc_margin']:.3f}",
            f"  {'':10s} later only:    rule raw {f(b['raw'])} | rule + threshold {f(b['thr'])} | logistic {f(b['model'])} | AUROC rule {b['auc_margin']:.3f}"]
txt = "\n".join(out); print(txt); open(os.path.join(HERE, "defcon_reset_fix_results.txt"), "w").write(txt + "\n")
