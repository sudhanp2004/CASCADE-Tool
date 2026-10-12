"""TV per assistant reply for Geometry (TV-only paper). Resumable + chunked so a session restart continues.
TV = gate(HarmBench+28) * (0.6 S_E[MiniLM] + 0.4 I_D[spaCy lexical density / .70]). Embeddings only, no reading.
Writes parts/part_XXXX.parquet (one per chunk); rerun resumes by skipping finished parts; combine_geo_tv() merges."""
import os, sys, glob
import numpy as np, pandas as pd, spacy
HERE = os.path.dirname(os.path.abspath(__file__)); SB = os.path.dirname(HERE); TT = os.path.join(SB, "tv_test")
sys.path.insert(0, TT)
from components import enc, s_e
nb = {"__file__": os.path.join(TT, "eval_baseline.py")}
exec(open(os.path.join(TT, "eval_baseline.py")).read().split("cas = pd.read_parquet")[0], nb); gate, FIX = nb["gate"], nb["FIX"]
PARTS = os.path.join(HERE, "parts"); os.makedirs(PARTS, exist_ok=True)
CHUNK = 4000
_nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
_nlp.max_length = 200000
CONTENT = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}
def i_d_batch(texts):
    out = []
    for doc in _nlp.pipe([t[:100000] for t in texts], batch_size=64):
        toks = [x for x in doc if not x.is_punct and not x.is_space]
        out.append(min(1.0, (sum(x.pos_ in CONTENT for x in toks) / max(len(toks), 1)) / 0.70) if toks else 0.0)
    return np.array(out)

def build():
    D = pd.read_parquet(os.path.join(SB, "datasets", "geometry", "final", "geometry_main.parquet"))
    cls_map = dict(zip(D.idx, D["class"])); rows = []
    for r in D.itertuples():
        reps = [m["content"] for m in r.messages if m["role"] == "assistant" and m["content"].strip()]
        for t, a in enumerate(reps):
            rows.append({"idx": r.idx, "model_name": r.model_name, "harm_turn": r.harm_turn,
                         "n_replies": r.n_replies, "objective": r.objective, "turn": t, "reply": a})
    R = pd.DataFrame(rows); R["cls"] = R.idx.map(cls_map)
    n = len(R); print("replies", n, flush=True)
    for c0 in range(0, n, CHUNK):
        pf = os.path.join(PARTS, f"part_{c0:06d}.parquet")
        if os.path.exists(pf): continue
        ch = R.iloc[c0:c0 + CHUNK].copy()
        E = enc(ch.reply.tolist())
        ch["tv"] = np.array([gate(t, FIX) for t in ch.reply]) * (.6 * np.asarray(s_e(E)) + .4 * i_d_batch(ch.reply.tolist()))
        ch.drop(columns=["reply"]).to_parquet(pf, index=False)
        print(f"chunk {c0}/{n} done", flush=True)
    parts = sorted(glob.glob(os.path.join(PARTS, "part_*.parquet")))
    pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True).to_parquet(os.path.join(HERE, "geo_tv.parquet"), index=False)
    print("DONE combined", len(parts), "parts", flush=True)

if __name__ == "__main__":
    build()
