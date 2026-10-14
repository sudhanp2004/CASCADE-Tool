#!/bin/bash
K=/home/sudhan/miniconda3/bin/kaggle; cd "$(dirname "$0")"
st(){ $K kernels status "$1" 2>&1; }
for i in $(seq 1 900); do s=$(st sudhanp2004/tv-sd-replay); echo "$s" | grep -qiE "complete|error|cancel" && break; sleep 30; done; echo "replay: $s"
mkdir -p out_replay ds_replay_out; $K kernels output sudhanp2004/tv-sd-replay -p out_replay >/dev/null 2>&1; ls -la out_replay; wc -l out_replay/replay.jsonl
cp out_replay/replay.jsonl ds_replay_out/; echo '{"title": "tv-sd-replay-out", "id": "sudhanp2004/tv-sd-replay-out", "licenses": [{"name": "other"}]}' > ds_replay_out/dataset-metadata.json
$K datasets create -p ds_replay_out 2>&1 | tail -1
for i in $(seq 1 30); do $K datasets status sudhanp2004/tv-sd-replay-out 2>&1 | grep -qi ready && break; sleep 10; done
$K kernels push -p kernel_judge 2>&1 | tail -1
for i in $(seq 1 1400); do s=$(st sudhanp2004/tv-sd-judge); echo "$s" | grep -qiE "complete|error|cancel" && break; sleep 30; done; echo "judge: $s"
mkdir -p out_judge; $K kernels output sudhanp2004/tv-sd-judge -p out_judge >/dev/null 2>&1; ls -la out_judge
