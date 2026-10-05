"""Full analysis -> explain_results.txt. Without a passed judge gate the output is marked PROVISIONAL
(user asked not to block on the hand check; the final version needs GATE: PASS)."""
import os
import pandas as pd
from frames import load_attack, load_benign
from measures import add_tv_pr
from analysis import (add_outcome, qualifying_goals, heldout_offset, side_shares, bootstrap_ci, paired_test,
                      lever_table, model_verdict, overall_verdict)

HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "data")

def main():
    gate = os.path.join(D, "judge_check.txt")
    passed = os.path.exists(gate) and "GATE: PASS" in open(gate).read()
    df = load_attack(); b = os.path.join(D, "benign_convs.jsonl")
    if os.path.exists(b): df = pd.concat([df, load_benign(b)], ignore_index=True)
    df = add_outcome(df.merge(pd.read_parquet(os.path.join(D, "labels.parquet")), on=["conv_key", "turn"]))
    df = add_tv_pr(df); df.to_parquet(os.path.join(D, "frame.parquet"), index=False)
    out = [("" if passed else "*** PROVISIONAL: judge hand-check gate not passed yet ***\n") +
           "TV JAILBREAK-EXPLANATION TEST (spec 2026-10-03). Offset comparison is primary; raw (c=0) in brackets.",
           "Outcome counts (all rows incl. no-prev):", df.groupby(["model", "outcome"]).size().unstack(fill_value=0).to_string(),
           "\nRows without a previous reply (excluded from P1-P4), outcome counts:",
           df[~df.has_prev].groupby(["model", "outcome"]).size().unstack(fill_value=0).to_string()]
    q = qualifying_goals(df); verdicts, levers = {}, []
    for m, g in df[df.has_prev].groupby("model"):
        g = g[g.outcome.isin(["refusal", "harmless", "jailbreak"])]
        c = heldout_offset(g); s = side_shares(g, c); s0 = side_shares(g, pd.Series(0.0, index=c.index))
        n, med, p = paired_test(g); lv = lever_table(g).assign(model=m); levers.append(lv)
        verdicts[m] = model_verdict(s, p, lv, q[m])
        out.append(f"\n== {m}  (qualifying goals {q[m]}; {verdicts[m]['status']})")
        for o in ("refusal", "harmless", "jailbreak"):
            lo, hi = bootstrap_ci(g, c, o)
            out.append(f"  P{('refusal', 'harmless', 'jailbreak').index(o) + 1} {o:9s} n={int((g.outcome == o).sum()):4d}  on predicted side {s[o]:.1%} [{lo:.1%}, {hi:.1%}]  (raw {s0[o]:.1%})")
        out.append(f"  P4 paired (jailbreak - refusal margin, same goal): goals={n}, median diff={med:+.3f}, p={p:.4f}")
        out.append(f"  P5 levers: {lv.groupby(['ref_type', 'cls']).size().to_dict()}")
        if verdicts[m]["status"] == "ok":
            out.append(f"  verdict: P1-P3 {verdicts[m]['P1_P3']}, P4 {verdicts[m]['P4']}, P5 {verdicts[m]['P5']} (neither {verdicts[m]['neither']:.1%})")
    pd.concat(levers).to_parquet(os.path.join(D, "levers.parquet"), index=False)
    hb = df[df.outcome == "harmful_benign"]
    out += [f"\nharmful_compliance under benign intent: {len(hb)} replies (reported, not a jailbreak by definition)",
            f"\nOVERALL: {overall_verdict(verdicts)}",
            "Note: a PASS supports the claim as a description of model behaviour; it does not show causation."]
    txt = "\n".join(out); print(txt); open(os.path.join(HERE, "explain_results.txt"), "w").write(txt + "\n")

if __name__ == "__main__":
    main()
