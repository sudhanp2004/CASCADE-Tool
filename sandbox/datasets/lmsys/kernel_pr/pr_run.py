"""PR for LMSYS requests: the paper's PR classifier retrained exactly (all-mpnet-base-v2 + LogisticRegressionCV on
WildGuardMix train prompts minus near-duplicates of evaluation texts, as build_request_risk.py), validated on
WildGuardMix test (must reproduce ~0.906), then applied to every LMSYS user message. Output /kaggle/working/pr_lmsys.parquet"""
import glob, json
import pandas as pd, torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
f = lambda n: glob.glob(f"/kaggle/input/**/{n}", recursive=True)[0]
G, TR, TE, RQ = (pd.read_parquet(f(n)) for n in ["guard.parquet", "wgm_train.parquet", "wgm_test.parquet", "requests.parquet"])
e = lambda m, t: m.encode(list(t), batch_size=256, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
mini = SentenceTransformer("all-MiniLM-L6-v2", device="cuda"); Eg, Et = torch.tensor(e(mini, G.text), device="cuda"), torch.tensor(e(mini, TR.text), device="cuda")
mx = torch.cat([(Et[i:i + 4096] @ Eg.T).max(1).values for i in range(0, len(Et), 4096)]).cpu().numpy()
keep = ~((mx >= .9) | TR.text.str.strip().str.lower().isin({g.lower() for g in G.text}).values); print("train kept", int(keep.sum()), flush=True)
m = SentenceTransformer("all-mpnet-base-v2", device="cuda")
lr = LogisticRegressionCV(Cs=[0.3, 1, 3, 10], cv=5, class_weight="balanced", scoring="roc_auc", max_iter=3000).fit(e(m, TR.text)[keep], TR.y.values[keep])
auc = float(roc_auc_score(TE.y, lr.predict_proba(e(m, TE.text))[:, 1])); print("WildGuardMix test AUROC", round(auc, 4), flush=True)
pd.DataFrame({"text": RQ.text, "pr": lr.predict_proba(e(m, RQ.text))[:, 1]}).to_parquet("/kaggle/working/pr_lmsys.parquet", index=False)
json.dump({"wgm_test_auroc": auc}, open("/kaggle/working/pr_lmsys_auc.json", "w")); print("DONE")
