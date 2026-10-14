#!/bin/bash
# Waits for Phi Kaggle jobs (battlecon/tv-sdp-a, sudhanp2004/tv-sdp-b), downloads outputs, runs the TV evaluation.
# Detached (setsid) so it survives a Claude session restart. Log: watch_phi.log
cd "$(dirname "$0")"; K=/home/sudhan/miniconda3/bin/kaggle; BC=$(cat ~/.kaggle/battlecon/access_token)
st() { if [ "$1" = a ]; then KAGGLE_API_TOKEN=$BC $K kernels status battlecon/tv-sdp-a 2>&1; else $K kernels status sudhanp2004/tv-sdp-b 2>&1; fi | grep -oE "KernelWorkerStatus\.[A-Z]+"; }
for s in a b; do
  until st $s | grep -qE "COMPLETE|ERROR|CANCEL"; do sleep 120; done
  echo "$(date) $s: $(st $s)"; rm -rf out_p$s; mkdir out_p$s
  if [ $s = a ]; then KAGGLE_API_TOKEN=$BC timeout 1800 $K kernels output battlecon/tv-sdp-a -p out_pa >/dev/null 2>&1
  else timeout 1800 $K kernels output sudhanp2004/tv-sdp-b -p out_pb >/dev/null 2>&1; fi
  ls out_p$s
done
if [ -f out_pa/judged.csv ] && [ -f out_pb/judged.csv ]; then VICTIM=phi /home/sudhan/miniconda3/bin/python sdl_eval.py 2>&1 | grep -v -i warn; else echo "MISSING judged.csv - check kernel logs"; fi
echo "WATCH DONE $(date)"
