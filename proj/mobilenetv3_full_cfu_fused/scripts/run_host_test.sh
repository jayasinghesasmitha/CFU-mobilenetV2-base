#!/bin/sh
# End-to-end PC test of the whole integration (no FPGA, no RISC-V toolchain needed):
#   real TFLM interpreter (this repo's vendored sources + the patched kernels in src/) + the real fused-CFU RTL
#   (cfu.v) clocked by Verilator.
#   1. the full MobileNetV3-small (5x5 depthwise, hard-swish, squeeze-and-excite) on the embedded image:
#      plain TFLM vs fused CFU, bit-identical outputs
#   2. synthetic models with 3x3/5x5 depthwise, hard-swish, explicit PAD, stride 1/2, squeeze-and-excite (both MEAN
#      quantisation paths, P > 56): plain TFLM vs fused CFU for several random weights / inputs
#
#   scripts/run_host_test.sh            VERIFY=1 scripts/run_host_test.sh   (also cross-check every block vs the C reference)
# Needs: g++, make, verilator (>= 5), python3 + numpy + 'pip install tflite flatbuffers'; the project placed at
# <CFU repo>/proj/mobilenetv3_full_cfu_fused  (or set TFLM=<third_party/tflite-micro> COMMON=<common/src>).
set -e
cd "$(dirname "$0")/.."
PROJ=$(pwd)
TFLM=${TFLM:-$(cd ../../third_party/tflite-micro && pwd)}
COMMON=${COMMON:-$(cd ../../common/src && pwd)}
EXTRA=""; [ -n "$VERIFY" ] && EXTRA="-DFUSED_VERIFY"
mkdir -p build_host
make -f sim/Makefile.host tflm TFLM=$TFLM COMMON=$COMMON EXTRA="$EXTRA"
(cd build_host && verilator -Wno-fatal -Wno-WIDTH -Wno-CASEINCOMPLETE -Wno-CASEOVERLAP --cc --top-module Cfu -O2 \
  --x-assign fast --x-initial fast --exe $PROJ/sim/host_mnv2.cpp $PROJ/build_host/libmnv2host.a \
  -CFLAGS "-std=c++14 -O1 -w -DFUSED_HOST -DTF_LITE_STATIC_MEMORY -DTF_LITE_DISABLE_X86_NEON -I$PROJ/src -I$PROJ/sim/host_overrides -I$TFLM -I$TFLM/third_party/gemmlowp -I$TFLM/third_party/flatbuffers/include -I$TFLM/third_party/ruy -I$TFLM/third_party/kissfft" \
  -LDFLAGS "-lm -lpthread" -Mdir vl --build -o host_mnv2 $PROJ/cfu.v > vl_host.log 2>&1)
H=./build_host/vl/host_mnv2
fail=0
check() { if cmp -s "$1" "$2"; then echo "   PASS  $3"; else echo "   FAIL  $3"; fail=1; fi; }

M=model/mobilenetv3_small_full_int8.tflite
echo "=== 1. full MobileNetV3-small (5x5 + hard-swish + squeeze-and-excite), embedded image ==="
$H $M --image inputs/cat.dat --fused 0 --out build_host/p.bin > /dev/null
$H $M --image inputs/cat.dat --fused 1 --out build_host/f.bin | grep -E "top1|Blocks run|CFU busy|Verification"
check build_host/p.bin build_host/f.bin "fused == plain (1000 INT8 outputs)"

echo "=== 2. synthetic models: 3x3/5x5, hard-swish, PAD, stride 1/2, squeeze-and-excite ==="
for ms in 7 3 11; do
  python3 tools/make_test_model.py build_host/syn_$ms.tflite $ms > /dev/null 2>&1
  python3 tools/gen_fused_plan.py build_host/syn_$ms.tflite build_host/plan_$ms.h build_host/plan_$ms.txt > /dev/null
  for r in 1 2; do
    $H build_host/syn_$ms.tflite --rand $r --fused 0 --plan build_host/plan_$ms.txt --out build_host/p.bin > /dev/null
    $H build_host/syn_$ms.tflite --rand $r --fused 1 --plan build_host/plan_$ms.txt --out build_host/f.bin > build_host/f.log
    check build_host/p.bin build_host/f.bin "model-seed $ms input-seed $r: $(grep 'Blocks run' build_host/f.log | sed 's/  */ /g')"
  done
done
[ $fail -eq 0 ] && echo "RESULT: PASS" || { echo "RESULT: FAIL"; exit 1; }
