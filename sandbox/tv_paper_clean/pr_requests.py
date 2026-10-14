"""Next user request after each locked screening reply (input for PR on Kaggle). No models.
JB: the harmful turn; safe: the final turn (both = user turn after reply screen_turn). -> locked_requests.parquet"""
import os, json, re, gzip
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); SB = os.path.dirname(HERE)
M = json.load(open(os.path.join(HERE, "locked", "manifest.json")))
T = pd.read_parquet(os.path.join(SB, "tv_gen/final/tv_gen.parquet")).set_index("idx")
Gm = pd.read_parquet(os.path.join(SB, "datasets/geometry/final/geometry_main.parquet")).set_index("idx")
raw = gzip.open(os.path.join(SB, "anthropic_full/red-team-attempts/red_team_attempts.jsonl.gz")).read().decode()
try: rows = json.loads(raw)
except json.JSONDecodeError: rows = [json.loads(l) for l in raw.splitlines() if l.strip()]
def next_request(ds, i, t):
    if ds == "Anthropic":
        hum = [s.split("\n\nAssistant:")[0].strip() for s in re.split(r"\n\nHuman:", rows[int(i)]["transcript"])[1:]]
        return hum[t + 1] if t + 1 < len(hum) else None
    msgs = list((T if ds == "tv_gen" else Gm).loc[i].messages)
    ai = [j for j, m in enumerate(msgs) if m["role"] == "assistant" and (ds == "tv_gen" or m["content"].strip())]
    nxt = [m["content"] for m in msgs[ai[t] + 1:] if m["role"] == "user"]
    return nxt[0] if nxt else None
out = []
for s_ in M["sets"]:
    L = pd.read_csv(os.path.join(HERE, "locked", s_["file"]))
    for r in L.itertuples():
        out.append({"file": s_["file"], "idx": str(r.idx), "screen_turn": r.screen_turn, "cls": r.cls, "tv": r.tv, "text": next_request(s_["dataset"], r.idx, r.screen_turn)})
D = pd.DataFrame(out); print(D.groupby("file").text.apply(lambda x: f"{x.notna().sum()}/{len(x)}").to_string())
D.to_parquet(os.path.join(HERE, "locked_requests.parquet"), index=False)
