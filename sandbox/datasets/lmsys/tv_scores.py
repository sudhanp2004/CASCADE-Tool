"""TV components for every LMSYS reply (same functions as the paper): gate (HarmBench list + 28 phrasings, tau .05,
gamma 2), S_E (MiniLM, Do-Not-Answer + CoCoNot anchors), I_D (spaCy lexical density / .70), hedge (Do-Not-Answer
p_scores only), log length. I_D is computed on the first 100,000 characters of a reply (spaCy limit is 1M; a few
LMSYS replies are millions of characters long; lexical density is a ratio, so truncation barely changes it). Output final/lmsys_reply_scores.parquet (text, gate, s_e, i_d, hedge, loglen)."""
import os, sys, json
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); TT = os.path.join(HERE, "..", "..", "tv_test"); sys.path.insert(0, TT)
from risk_signal import enc, p_scores
from components import s_e, i_d
nb = {"__file__": os.path.join(TT, "eval_baseline.py")}
exec(open(os.path.join(TT, "eval_baseline.py")).read().split("cas = pd.read_parquet")[0], nb); gate, FIX = nb["gate"], nb["FIX"]
reps = sorted({json.loads(l)["response"] for l in open(os.path.join(HERE, "wg_ds", "pairs.jsonl"))})
print("unique replies", len(reps), flush=True)
E = enc(reps); print("embedded", flush=True)
out = pd.DataFrame({"text": reps, "gate": [gate(t, FIX) for t in reps], "s_e": s_e(E), "hedge": p_scores(E), "loglen": [np.log1p(len(t.split())) for t in reps]})
out.to_parquet(os.path.join(HERE, "final", "lmsys_reply_scores_partial.parquet"), index=False)
print("gate/s_e/hedge done; replies over 100k chars:", sum(len(t) > 100000 for t in reps), flush=True)
out["i_d"] = [i_d(t[:100000]) for t in reps]
out.to_parquet(os.path.join(HERE, "final", "lmsys_reply_scores.parquet"), index=False); print("DONE", len(out))
