"""One row per assistant reply, same schema for CASCADE, tv_gen and benign-twin conversations."""
import ast, json, os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CASCADE = os.path.join(HERE, "..", "datasets", "cascade", "final", "cascade.parquet")
TVGEN = os.path.join(HERE, "..", "tv_gen", "final", "tv_gen.parquet")
COLS = ["conv_key", "source", "model", "goal", "goal_id", "intent", "turn", "user", "reply", "jbb"]

def _labels(v):
    if isinstance(v, str):
        return [int(x) for x in ast.literal_eval(v.replace(" ", ","))]
    return [int(x) for x in v]

def from_messages(rows):
    out = []
    for r in rows:
        A = [m["content"] for m in r["messages"] if m["role"] == "assistant"]
        U = [m["content"] for m in r["messages"] if m["role"] == "user"]
        if len(r["jbb"]) != len(A):
            raise ValueError(f"{r['conv_key']}: {len(r['jbb'])} labels for {len(A)} replies")
        for k, (u, a, y) in enumerate(zip(U, A, r["jbb"])):
            out.append({**{c: r[c] for c in COLS[:6]}, "turn": k, "user": u, "reply": a or "", "jbb": int(y)})
    return pd.DataFrame(out, columns=COLS)

def load_attack(cascade_path=CASCADE, tvgen_path=TVGEN):
    C, T = pd.read_parquet(cascade_path), pd.read_parquet(tvgen_path)
    rows = [{"conv_key": f"cascade:{r.session_id}", "source": "cascade", "model": r.target_model, "goal": r.goal,
             "goal_id": r.goal, "intent": "harmful", "messages": list(r.messages), "jbb": _labels(r.jbb_judge_per_reply)}
            for r in C.itertuples()]
    rows += [{"conv_key": f"tv_gen:{r.idx}", "source": "tv_gen", "model": r.target, "goal": r.goal, "goal_id": r.goal,
              "intent": "harmful", "messages": list(r.messages), "jbb": _labels(r.jbb_per_reply)} for r in T.itertuples()]
    return from_messages(rows)

def load_benign(jsonl_path):
    rows = []
    for line in open(jsonl_path):
        r = json.loads(line)
        msgs = [m for t in r["turns"] for m in ({"role": "user", "content": t["user"]}, {"role": "assistant", "content": t["target"]})]
        rows.append({"conv_key": f"benign:{r['goal_index']}:{r['target']}:{r['attempt']}", "source": "benign",
                     "model": r["target"], "goal": r["benign_goal"], "goal_id": r["harmful_goal"], "intent": "benign",
                     "messages": msgs, "jbb": [-1] * len(r["turns"])})
    return from_messages(rows)

def prior_pairs(df):
    out, last_key, acc = [], None, []
    for r in df.itertuples():
        if r.conv_key != last_key:
            last_key, acc = r.conv_key, []
        out.append(list(acc)); acc.append((r.user, r.reply))
    return out
