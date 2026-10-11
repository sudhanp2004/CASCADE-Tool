"""PR computed on THIS PC (CPU), identical method to the Kaggle kernel tv-pr-* (all-mpnet-base-v2 + LogisticRegressionCV
on WildGuardMix train prompts minus near-duplicates of evaluation texts; validate on WildGuardMix test ~0.906).
Makes the perceived-risk step fully local instead of Kaggle-only. Inputs already in the repo:
  sandbox/tv_pr_ctx/ds/{wgm_train,wgm_test,guard}.parquet   WildGuardMix train/test + near-dup reference
  sandbox/tv_pr_ctx/ds/ctx.parquet                          request text ('cur') keyed to the tv_gen/DEF CON frames
Outputs sandbox/clean/pr_local.parquet (key, text, pr) and prints: test AUROC, and agreement with the frozen pr_msg
already baked into the locked frame_*.parquet (should match the Kaggle values). LMSYS requests are scored too if
sandbox/datasets/lmsys/pr_ds/requests.parquet is present."""
import glob, json, os
import numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__)); SB = os.path.dirname(HERE); DS = os.path.join(SB, "tv_pr_ctx", "ds")
G, TR, TE = (pd.read_parquet(os.path.join(DS, n)) for n in ("guard.parquet", "wgm_train.parquet", "wgm_test.parquet"))
enc = lambda m, t: m.encode(list(t), batch_size=128, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)

# near-duplicate removal, exactly as the kernel (MiniLM cos >= .9 to any eval/guard text, or exact match), on CPU
mini = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
Eg, Et = enc(mini, G.text), enc(mini, TR.text)
mx = np.concatenate([(Et[i:i + 4096] @ Eg.T).max(1) for i in range(0, len(Et), 4096)])
keep = ~((mx >= .9) | TR.text.str.strip().str.lower().isin({g.lower() for g in G.text}).values)
print("train kept", int(keep.sum()), "of", len(TR), flush=True)

mp = SentenceTransformer("all-mpnet-base-v2", device="cpu")
lr = LogisticRegressionCV(Cs=[0.3, 1, 3, 10], cv=5, class_weight="balanced", scoring="roc_auc", max_iter=3000).fit(enc(mp, TR.text)[keep], TR.y.values[keep])
auc = float(roc_auc_score(TE.y, lr.predict_proba(enc(mp, TE.text))[:, 1]))
print(f"WildGuardMix test AUROC {auc:.4f}  (Kaggle/paper: 0.906)", flush=True)

# score the locked tv_gen / DEF CON requests and compare to the frozen pr_msg
C = pd.read_parquet(os.path.join(DS, "ctx.parquet"))
rows = []
for f in glob.glob(os.path.join(HERE, "frame_*.parquet")):
    g = pd.read_parquet(f)
    if "ckey" not in g: continue                      # LMSYS frames carry no ckey; handled separately
    m = C.set_index("key").cur.reindex(g.ckey.values)
    sub = pd.DataFrame({"key": g.ckey.values, "text": m.values, "frozen": g.PR.values}).dropna()
    rows.append((os.path.basename(f), sub))
if rows:
    allt = pd.concat([s for _, s in rows]); uniq = allt.drop_duplicates("text").reset_index(drop=True)
    pr = lr.predict_proba(enc(mp, uniq.text))[:, 1]; pmap = dict(zip(uniq.text, pr))
    allt["pr_local"] = allt.text.map(pmap)
    allt.to_parquet(os.path.join(HERE, "pr_local.parquet"), index=False)
    print("\nAGREEMENT local PR vs frozen Kaggle pr_msg (per locked frame):")
    for name, s in rows:
        s = s.copy(); s["pr_local"] = s.text.map(pmap)
        print(f"  {name[:40]:40s} n={len(s):5d} corr={np.corrcoef(s.frozen, s.pr_local)[0,1]:.4f} max|diff|={np.abs(s.frozen - s.pr_local).max():.3f}")

rq = os.path.join(SB, "datasets", "lmsys", "pr_ds", "requests.parquet")
if os.path.exists(rq):
    R = pd.read_parquet(rq); R["pr"] = lr.predict_proba(enc(mp, R.text))[:, 1]
    R.to_parquet(os.path.join(HERE, "pr_local_lmsys.parquet"), index=False); print(f"\nLMSYS requests scored: {len(R)} -> pr_local_lmsys.parquet")
json.dump({"wgm_test_auroc": auc, "train_kept": int(keep.sum())}, open(os.path.join(HERE, "pr_local_auc.json"), "w"))
print("DONE")
