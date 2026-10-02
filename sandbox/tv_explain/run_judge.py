"""Judge every reply (attack frames, or benign with --benign) -> data/labels.parquet (merged, resumable via cache)."""
import os, sys
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from groq_client import GroqClient, load_keys
from judge3 import judge_reply
from frames import load_attack, load_benign, prior_pairs
from analysis import add_outcome, qualifying_goals

HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "data")
OUT = os.path.join(D, "labels.parquet")

def main(benign):
    df = load_benign(os.path.join(D, "benign_convs.jsonl")) if benign else load_attack()
    pri = prior_pairs(df)
    client = GroqClient(load_keys(), os.path.join(D, "cache.jsonl"), rounds=10)
    jobs = list(zip(df.goal, pri, df.user, df.reply))
    with ThreadPoolExecutor(8) as ex:
        labels = []
        for i, lab in enumerate(ex.map(lambda j: judge_reply(client, *j), jobs)):
            labels.append(lab)
            if i % 200 == 0: print(f"{i}/{len(jobs)}", flush=True)
    new = pd.DataFrame({"conv_key": df.conv_key, "turn": df.turn, "label": labels})
    old = pd.read_parquet(OUT) if os.path.exists(OUT) else new.iloc[0:0]
    allv = pd.concat([old[~old.conv_key.isin(set(new.conv_key))], new]).reset_index(drop=True)
    allv.to_parquet(OUT, index=False)
    d = add_outcome(df.merge(new, on=["conv_key", "turn"]))
    print(d.groupby(["model", "outcome"]).size().unstack(fill_value=0).to_string())
    if not benign:
        q = qualifying_goals(d)
        txt = "Qualifying goals per model (>=1 refusal and >=1 jailbreak at turn>=1):\n" + "\n".join(f"  {m}: {n}" for m, n in q.items())
        print(txt); open(os.path.join(D, "step1_counts.txt"), "w").write(txt + "\n")

if __name__ == "__main__":
    main("--benign" in sys.argv)
