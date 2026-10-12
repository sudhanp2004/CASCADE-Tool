"""Early-warning test of TV alone (V3 TV-paper claim), per target model, on the cleaned Geometry agreement subset.
Claim: above the TV threshold the model is MORE LIKELY to comply with the next harmful request; below => stays safe.
Operationalisation (screening test, per conversation):
  jailbroken conv -> screening point = TV at the reply JUST BEFORE the first harmful reply (turn harm_turn-1). Positive.
                     (harm_turn==0 dropped: harm at reply 1, no previous reply to screen.)
  safe conv (failed attack, hard negative) -> TV = max over its replies (the screen 'fires' if TV ever exceeds c). Negative.
Per model: AUROC (N-1 TV on JB vs max TV on safe), bootstrap CI over objectives; and an operating point with the
threshold set to a 10% false-alarm rate on safe conversations -> sensitivity (JB caught before harm). Grouped by objective."""
import os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__))
R = pd.read_parquet(os.path.join(HERE, "geo_tv.parquet"))
out = ["EARLY-WARNING TEST — TV alone, Geometry agreement subset, per target model",
       "screen point: JB = TV one reply before first harm (N-1); safe = max TV over conv (hard negatives = failed attacks)", ""]
out.append(f"  {'model':22s} {'JB(N-1)':>7s} {'safe':>6s} | {'AUROC [95% CI]':>22s} | {'sens @10% false-alarm':>22s}")
rng = np.random.default_rng(0)
for m, g in R.groupby("model_name"):
    jb = g[g.cls == "jailbroken"]; sf = g[g.cls == "safe"]
    # JB: TV at harm_turn-1, per conversation (harm_turn>=1)
    pos = []
    for idx, c in jb.groupby("idx"):
        ht = int(c.harm_turn.iloc[0])
        if ht < 1: continue
        v = c[c.turn == ht - 1].tv
        if len(v): pos.append((c.objective.iloc[0], float(v.iloc[0])))
    # safe: max TV over the conversation
    neg = [(o, float(cg.tv.max())) for (idx, o), cg in sf.groupby(["idx", "objective"])]
    P = pd.DataFrame(pos, columns=["obj", "tv"]); N = pd.DataFrame(neg, columns=["obj", "tv"])
    y = np.r_[np.ones(len(P)), np.zeros(len(N))]; s = np.r_[P.tv.values, N.tv.values]
    auc = roc_auc_score(y, s)
    thr = np.quantile(N.tv.values, 0.90); sens = (P.tv.values > thr).mean()
    # bootstrap over objectives for AUROC CI
    objs = pd.concat([P.obj, N.obj]).unique(); byP = {o: P[P.obj == o] for o in P.obj.unique()}; byN = {o: N[N.obj == o] for o in N.obj.unique()}
    bs = []
    for _ in range(1000):
        ch = rng.choice(objs, len(objs))
        pp = pd.concat([byP[o] for o in ch if o in byP]); nn = pd.concat([byN[o] for o in ch if o in byN])
        if len(pp) < 2 or len(nn) < 2: continue
        yy = np.r_[np.ones(len(pp)), np.zeros(len(nn))]; ss = np.r_[pp.tv.values, nn.tv.values]
        if len(np.unique(yy)) == 2: bs.append(roc_auc_score(yy, ss))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    out.append(f"  {m[:22]:22s} {len(P):7d} {len(N):6d} | {auc:.3f} [{lo:.3f}, {hi:.3f}] | {sens:21.0%}")
txt = "\n".join(out); print(txt); open(os.path.join(HERE, "early_warning_results.txt"), "w").write(txt + "\n")
