"""Early-warning test of TV alone on Anthropic red-team (human attackers, refusals kept -> fits TV).
Conversations = datasets/anthropic/candidates.parquet (StrongREJECT per-reply judge; first_harm = first harmful reply).
TV per reply = gate(HarmBench+28) * (0.6 S_E + 0.4 I_D), embeddings only.
Screen point: jailbroken -> TV at reply first_harm-1 (first_harm>=1); safe -> replies at the same turn positions.
Per model type (and the 400 hand-checked 'kept' subset): AUROC, sensitivity at 10% false alarm."""
import os, sys, json, gzip, re
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__)); SB = os.path.dirname(HERE); TT = os.path.join(SB, "tv_test")
sys.path.insert(0, TT)
from components import enc, s_e, i_d
nb = {"__file__": os.path.join(TT, "eval_baseline.py")}
exec(open(os.path.join(TT, "eval_baseline.py")).read().split("cas = pd.read_parquet")[0], nb); gate, FIX = nb["gate"], nb["FIX"]
OUT = os.path.join(HERE, "anth_tv.parquet")
if not os.path.exists(OUT):
    raw = gzip.open(os.path.join(SB, "anthropic_full/red-team-attempts/red_team_attempts.jsonl.gz")).read().decode()
    try: rows = json.loads(raw)
    except json.JSONDecodeError: rows = [json.loads(l) for l in raw.splitlines() if l.strip()]
    C = pd.read_parquet(os.path.join(SB, "datasets/anthropic/candidates.parquet"))
    keep = set(pd.read_csv(os.path.join(SB, "datasets/anthropic/decisions.tsv"), sep="\t").query("verdict=='keep'").idx)
    rec = []
    for r in C.itertuples():
        reps = [s.split("\n\nHuman:")[0].strip() for s in re.split(r"\n\nAssistant:", rows[int(r.idx)]["transcript"])[1:]]
        for t, a in enumerate(reps):
            rec.append({"idx": r.idx, "cls": r.label, "first_harm": r.first_harm, "model": f"{r.model_type} {r.params}",
                        "kept": r.idx in keep, "turn": t, "reply": a})
    R = pd.DataFrame(rec); print("replies", len(R), flush=True)
    E = enc(R.reply.tolist())
    R["tv"] = np.array([gate(t, FIX) for t in R.reply]) * (.6 * np.asarray(s_e(E)) + .4 * np.array([i_d(t[:100000]) for t in R.reply]))
    R.drop(columns=["reply"]).to_parquet(OUT, index=False)
R = pd.read_parquet(OUT)
def ew(g, name):
    jb = g[(g.cls == "jailbroken") & (g.first_harm >= 1)]; P = jb[jb.turn == jb.first_harm - 1]
    S = g[(g.cls == "safe") & (g.turn.isin(P.turn.unique()))]
    if len(P) < 15 or len(S) < 15: return f"  {name:32s} too few (JB {len(P)}, safe {len(S)})"
    y = np.r_[np.ones(len(P)), np.zeros(len(S))]; s = np.r_[P.tv, S.tv]; thr = np.quantile(S.tv, .9)
    return f"  {name:32s} JB {len(P):4d} safe-pts {len(S):5d} | AUROC {roc_auc_score(y, s):.3f} | sens@10%FA {(P.tv > thr).mean():4.0%} | mean TV JB {P.tv.mean():.3f} safe {S.tv.mean():.3f}"
out = ["EARLY WARNING — TV alone, Anthropic red-team (turn-matched)", ew(R, "ALL candidates"), ew(R[R.kept], "hand-checked 400")]
for m, g in R.groupby("model"): out.append(ew(g, m))
txt = "\n".join(out); print(txt); open(os.path.join(HERE, "anthropic_ew_results.txt"), "w").write(txt + "\n")
