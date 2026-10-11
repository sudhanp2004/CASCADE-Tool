"""LMSYS-Chat-1M sample for the trust-risk theory (per LLM, real users, real replies).
Selection (fixed before any labelling): English, >= 2 user turns, models with >= 80 conversations that contain an
OpenAI-moderation-flagged USER message. Per model: all flagged conversations (capped at 500, random) + the same number
of unflagged conversations (random), seed 0. Every (user message, assistant reply) pair with a non-empty reply is a
decision. Outputs:
  final/lmsys_sample.parquet  (conversation_id, model, flagged, messages)
  wg_ds/pairs.jsonl           (ds=lmsys, row, prompt, response) for WildGuard 7B labelling
  pr_ds/requests.parquet      (text) every user message, for PR scoring"""
import glob, json, os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ["final", "wg_ds", "pr_ds"]: os.makedirs(os.path.join(HERE, d), exist_ok=True)
D = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(HERE, "raw", "*.parquet")))], ignore_index=True)
D = D[(D.language == "English") & (D.turn >= 2)].copy()
D["flagged"] = D.openai_moderation.map(lambda ms: any(m["flagged"] for i, m in enumerate(ms) if i % 2 == 0))
keep = D[D.flagged].groupby("model").size(); keep = keep[keep >= 80].index
parts = []
for m in keep:
    g = D[D.model == m]; f = g[g.flagged]; f = f.sample(min(len(f), 500), random_state=0)
    u = g[~g.flagged].sample(min(len(f), (~g.flagged).sum()), random_state=0)
    parts += [f, u]
S = pd.concat(parts)[["conversation_id", "model", "flagged", "conversation"]].rename(columns={"conversation": "messages"}).reset_index(drop=True)
S["messages"] = S.messages.map(lambda ms: [{"role": m["role"], "content": m["content"]} for m in ms])
S.to_parquet(os.path.join(HERE, "final", "lmsys_sample.parquet"), index=False)

rows, reqs = [], set()
for r in S.itertuples():
    last = None
    for m in r.messages:
        if m["role"] == "user": last = m["content"]; reqs.add(m["content"])
        elif m["role"] == "assistant" and last is not None and m["content"].strip():
            rows.append({"ds": "lmsys", "row": len(rows), "prompt": last, "response": m["content"]}); last = None
with open(os.path.join(HERE, "wg_ds", "pairs.jsonl"), "w") as fh:
    for x in rows: fh.write(json.dumps(x) + "\n")
pd.DataFrame({"text": sorted(reqs)}).to_parquet(os.path.join(HERE, "pr_ds", "requests.parquet"), index=False)
print("models", len(keep), "| conversations", len(S), "| decisions (pairs)", len(rows), "| unique requests", len(reqs))
print(S.groupby(["model", "flagged"]).size().unstack().to_string())
