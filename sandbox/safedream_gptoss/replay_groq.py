"""SafeDream datasets on gpt-oss-20b (open-weight, served by the Groq API = API-only victim, no hidden states).
Same 1,500 conversations as the Qwen/Llama/Phi replays (safedream_h2h/replay_in.jsonl). Turn by turn, the model sees its
own replies. Settings identical to our tv_gen gpt-oss-20b target: temperature 0.7, max_tokens 1024 (+512 reasoning
budget), reasoning_effort low (tv_explain/groq_client.py, 13 rotating keys).
FAIL-SAFE: every API reply is cached (groq_cache.jsonl) and each finished conversation is appended to out/replay.jsonl;
a rerun skips finished conversations and replays cached turns instantly. Conversations whose keys are exhausted are
listed in failed.txt and retried on the next run. Run detached: setsid nohup python replay_groq.py &"""
import json, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__)); SB = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(SB, "tv_explain"))
from groq_client import GroqClient, GroqExhausted, load_keys
MODEL = "openai/gpt-oss-20b"; OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
RP = os.path.join(OUT, "replay.jsonl")
C = [json.loads(l) for l in open(os.path.join(SB, "safedream_h2h", "replay_in.jsonl"))]
done = {(r["ds"], str(r["id"])) for r in map(json.loads, open(RP))} if os.path.exists(RP) else set()
todo = [c for c in C if (c["ds"], str(c["id"])) not in done]
print(f"{len(done)} done, {len(todo)} to do", flush=True)
g = GroqClient(load_keys(), os.path.join(HERE, "groq_cache.jsonl"), rounds=20, round_wait=60)
lock = threading.Lock(); failed = []; n_ok = [0]; T0 = time.time()
def run(c):
    msgs, reps = [], []
    try:
        for u in c["user_turns"]:
            msgs.append({"role": "user", "content": u})
            r = g.chat(MODEL, msgs, max_tokens=1024, temperature=0.7, salt="sd-gptoss")
            reps.append(r); msgs.append({"role": "assistant", "content": r})
    except GroqExhausted:
        with lock: failed.append(f"{c['ds']}:{c['id']}")
        return
    with lock:
        with open(RP, "a") as f: f.write(json.dumps({"ds": c["ds"], "id": c["id"], "goal": c["goal"], "user_turns": c["user_turns"], "replies": reps}) + "\n")
        n_ok[0] += 1
        if n_ok[0] % 25 == 0: print(f"{n_ok[0]}/{len(todo)} convs, {(time.time()-T0)/60:.0f} min", flush=True)
with ThreadPoolExecutor(12) as ex: list(ex.map(run, todo))
open(os.path.join(HERE, "failed.txt"), "w").write("\n".join(failed) + ("\n" if failed else ""))
total = sum(1 for _ in open(RP)) if os.path.exists(RP) else 0
print(f"FINISHED this run: ok {n_ok[0]}, failed {len(failed)} | replay.jsonl has {total}/{len(C)}", flush=True)
