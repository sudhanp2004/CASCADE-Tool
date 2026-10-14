"""SafeDream head-to-head, step 1: replay published user turns (XGuard-Train, SafeDialBench, SafeMTData_1K;
500 conversations each, seed 0) on Qwen2.5-7B-Instruct (SafeDream's target), turn by turn; the model sees its OWN replies.
Kaggle's image is Python 3.13, where vllm 0.6.6.post1 cannot install, so we build a Python 3.11 venv with uv and run the
same engine as our GOAT run: vLLM 0.6.6.post1 fp16 TP2 on 2xT4; temp .7, top_p .8, top_k 20, rep 1.05, max 1024, seed 0.
SMOKE=True -> 5 conversations x 2 turns only (environment check). Output /kaggle/working/replay.jsonl."""
import os, subprocess, sys, textwrap
SMOKE = False
W = "/kaggle/working"
def sh(c): print("$", c, flush=True); subprocess.run(c, shell=True, check=True)
sh(f"{sys.executable} -m pip install -q uv")
sh("uv python install 3.11")
sh(f"uv venv -p 3.11 /tmp/venv")
sh(f"uv pip install -p /tmp/venv/bin/python vllm==0.6.6.post1 transformers==4.47.1 huggingface_hub setuptools")
gen = textwrap.dedent(f'''
    import glob, json
    from huggingface_hub import snapshot_download, HfApi
    from vllm import LLM, SamplingParams
    MODEL = "Qwen/Qwen2.5-7B-Instruct"; rev = HfApi().model_info(MODEL).sha; path = snapshot_download(MODEL, revision=rev)
    print("model rev", rev, flush=True)
    C = [json.loads(l) for l in open(glob.glob("/kaggle/input/**/replay_in.jsonl", recursive=True)[0])]
    if {SMOKE}: C = C[:5]
    llm = LLM(model=path, dtype="float16", tensor_parallel_size=2, max_model_len=32768, gpu_memory_utilization=0.90, seed=0)
    sp = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05, max_tokens=1024, seed=0)
    for c in C: c["replies"] = []
    T = 2 if {SMOKE} else max(len(c["user_turns"]) for c in C)
    for t in range(T):
        act = [c for c in C if len(c["user_turns"]) > t]
        msgs = []
        for c in act:
            m = []
            for k in range(t):
                m += [{{"role": "user", "content": c["user_turns"][k]}}, {{"role": "assistant", "content": c["replies"][k]}}]
            msgs.append(m + [{{"role": "user", "content": c["user_turns"][t]}}])
        outs = llm.chat(msgs, sp, use_tqdm=False)
        for c, o in zip(act, outs): c["replies"].append(o.outputs[0].text)
        print(f"turn {{t}}: {{len(act)}} generated", flush=True)
        with open("{W}/replay.jsonl", "w") as f:
            for c in C: f.write(json.dumps(c) + "\\n")
    json.dump({{"model": MODEL, "revision": rev, "smoke": {SMOKE}}}, open("{W}/run_info.json", "w")); print("DONE", flush=True)
''')
open(f"{W}/gen.py", "w").write(gen)
sh(f"/tmp/venv/bin/python {W}/gen.py")
