#!/bin/sh
# PC-side tests of the webcam pipeline (needs build_host/vl/host_mnv2, see scripts/run_host_test.sh):
#  1. image conversion is byte-identical to the existing cat_image.dat
#  2. one picture through the PC program and the simulated board (real TFLM + real RTL)
#  3. three different pictures: fused outputs == plain TFLM outputs (all 1000 values)
set -e
cd "$(dirname "$0")/../pc"
T=$(mktemp -d)
python3 webcam_classifier.py --no-send --image ../inputs/cat.jpg --count 1 --first-delay 0 --out-dir $T/conv >/dev/null
cmp $T/conv/*.dat ../inputs/cat.dat && echo "1. conversion identical to cat_image.dat: OK"
python3 webcam_classifier.py --sim --image ../inputs/cat.jpg --count 1 --first-delay 0 --no-preview --out-dir $T/one | grep -E "TOP-1"
mkdir $T/imgs; cp ../inputs/cat.jpg $T/imgs/a.jpg
python3 - <<PY
import numpy as np, cv2
rng=np.random.default_rng(3)
cv2.imwrite("$T/imgs/b.png", rng.integers(0,255,(300,400,3),dtype=np.uint8))
cv2.imwrite("$T/imgs/c.png", np.tile(np.linspace(0,255,640,dtype=np.uint8),(480,1))[:,:,None].repeat(3,2))
PY
python3 webcam_classifier.py --sim --image-dir $T/imgs --count 3 --interval 1 --first-delay 0 --no-preview --out-dir $T/f >/dev/null
python3 webcam_classifier.py --sim --sim-plain --image-dir $T/imgs --count 3 --interval 1 --first-delay 0 --no-preview --out-dir $T/p >/dev/null
python3 - <<PY
import json,glob
F=sorted(glob.glob("$T/f/*.json")); P=sorted(glob.glob("$T/p/*.json"))
ok=all(json.load(open(a))["logits"]==json.load(open(b))["logits"] for a,b in zip(F,P)) and len(F)==3
print("3. fused == plain on 3 pictures:", "OK" if ok else "MISMATCH"); raise SystemExit(0 if ok else 1)
PY
