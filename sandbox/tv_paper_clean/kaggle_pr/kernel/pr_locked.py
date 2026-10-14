"""PR for the next request after each LOCKED TV-only screening reply. PR classifier retrained exactly as the paper /
sandbox/clean/build_pr.py (all-mpnet-base-v2 + LogisticRegressionCV on WildGuardMix train prompts minus near-duplicates
of evaluation texts: MiniLM cos >= .9 or exact; validated on WildGuardMix test). Saves EVERYTHING reusable:
pr_model.joblib (lr, keep), pr_lr_params.npz (coef/intercept, sklearn-version-free), embeddings (float16 .npy),
locked_pr.parquet (file, idx, pr, leak_cos = max MiniLM cos of the request to a kept WildGuardMix train prompt)."""
import glob, json
import numpy as np, pandas as pd, torch, joblib
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
W = "/kaggle/working"
f = lambda n: glob.glob(f"/kaggle/input/**/{n}", recursive=True)[0]
G, TR, TE, R = (pd.read_parquet(f(n)) for n in ["guard.parquet", "wgm_train.parquet", "wgm_test.parquet", "locked_requests.parquet"])
R = R[R.text.notna()].reset_index(drop=True)
e = lambda m, t: m.encode(list(t), batch_size=256, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
mini = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
Eg, Et, Er = e(mini, G.text), e(mini, TR.text), e(mini, R.text)
np.save(f"{W}/mini_train.npy", Et.astype(np.float16)); np.save(f"{W}/mini_guard.npy", Eg.astype(np.float16))
tg, tt = torch.tensor(Eg, device="cuda"), torch.tensor(Et, device="cuda")
mx = torch.cat([(tt[i:i + 4096] @ tg.T).max(1).values for i in range(0, len(tt), 4096)]).cpu().numpy()
keep = ~((mx >= .9) | TR.text.str.strip().str.lower().isin({g.lower() for g in G.text}).values)
print("train kept", int(keep.sum()), "dropped", int((~keep).sum()), flush=True)
mp = SentenceTransformer("all-mpnet-base-v2", device="cuda")
Mt, Mte, Mr = e(mp, TR.text), e(mp, TE.text), e(mp, R.text)
np.save(f"{W}/mpnet_train.npy", Mt.astype(np.float16)); np.save(f"{W}/mpnet_test.npy", Mte.astype(np.float16))
lr = LogisticRegressionCV(Cs=[0.3, 1, 3, 10], cv=5, class_weight="balanced", scoring="roc_auc", max_iter=3000, n_jobs=-1).fit(Mt[keep], TR.y.values[keep])
joblib.dump((lr, keep), f"{W}/pr_model.joblib"); np.savez(f"{W}/pr_lr_params.npz", coef=lr.coef_, intercept=lr.intercept_, C=lr.C_, keep=keep)
auc = float(roc_auc_score(TE.y, lr.predict_proba(Mte)[:, 1])); print("WildGuardMix test AUROC", round(auc, 4), flush=True)
R["pr"] = lr.predict_proba(Mr)[:, 1]
tk = torch.tensor(Et[keep], device="cuda"); tr_ = torch.tensor(Er, device="cuda")
R["leak_cos"] = torch.cat([(tr_[i:i + 1024] @ tk.T).max(1).values for i in range(0, len(tr_), 1024)]).cpu().numpy()
R.drop(columns=["text"]).to_parquet(f"{W}/locked_pr.parquet", index=False)
json.dump({"wgm_test_auroc": auc, "train_kept": int(keep.sum()), "C": float(lr.C_[0])}, open(f"{W}/pr_info.json", "w"))
print("DONE", flush=True)
