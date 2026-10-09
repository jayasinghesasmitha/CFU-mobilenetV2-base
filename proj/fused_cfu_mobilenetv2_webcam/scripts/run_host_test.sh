#!/bin/sh
# End-to-end PC test of the whole integration (no FPGA, no RISC-V toolchain needed):
#   real TFLM interpreter (this repo's vendored sources + the patched kernels in src/)
#   + the real fused-CFU RTL (cfu.v) clocked by Verilator
# It runs MobileNetV2 on the embedded cat image twice (plain TFLM, then with the fused CFU) and checks that the
# 1000 FLOAT32 outputs are bit-identical.
#
#   scripts/run_host_test.sh            # plain vs fused
#   VERIFY=1 scripts/run_host_test.sh   # additionally cross-check every fused block against the C reference
#
# Needs: g++, make, verilator (>= 5.0), and this project placed at <CFU repo>/proj/fused_cfu_with_mobilenetv2
# (or set TFLM=<path to third_party/tflite-micro> COMMON=<path to common/src>).
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
M=model/mobilenetv2_a035_224_int8.tflite
echo "=== plain TFLM (no CFU) ==="; ./build_host/vl/host_mnv2 $M --fused 0 --image inputs/cat.dat --out build_host/plain.bin | grep -E "mode=|FNV"
echo "=== fused CFU (real RTL) ==="; ./build_host/vl/host_mnv2 $M --fused 1 --image inputs/cat.dat --out build_host/fused.bin | grep -vE "^CFU PING|^input type"
if cmp -s build_host/plain.bin build_host/fused.bin; then echo; echo "RESULT: PASS - fused-CFU output is bit-identical to plain TFLM"; else echo; echo "RESULT: FAIL - outputs differ"; exit 1; fi
