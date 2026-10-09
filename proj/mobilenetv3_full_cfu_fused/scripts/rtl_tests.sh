#!/bin/sh
# RTL block tests (PC): the real C driver (dsc_run_block) -> instruction log -> real RTL (Verilator) -> responses
# -> driver reassembles the output -> compared with the Python golden model.
# Covers 3x3 and 5x5 depthwise, stride 1/2, odd sizes, tiny maps, row-strip tiling, projection channel groups,
# many input-channel chunks and the activation tables.   Needs: verilator, gcc, python3 + numpy.
set -e
cd "$(dirname "$0")/.."
if [ ! -x build_tb/obj_dir/simv ]; then
  mkdir -p build_tb
  (cd build_tb && verilator --binary --timing -Wno-fatal -Wno-WIDTH -Wno-CASEINCOMPLETE -Wno-CASEOVERLAP \
     --top-module tb_replay -o simv ../sim/tb_replay.v ../cfu.v > vl.log 2>&1)
fi
fail=0
t() { echo "--- $1   limits=[$2]"
      out=$(./scripts/block_test.sh "$1" $2 2>&1 | grep -E "\[check\]"); echo "$out"
      echo "$out" | grep -q "PASS" || fail=1
      echo "$out" | grep -q "DIFFERENT" && fail=1 || true
      echo "$out" | grep -q "squeeze sums: FAIL" && fail=1 || true; }
# --- 3x3 (unchanged behaviour)
t "--H 8 --W 8 --N 8 --M 16 --P 8 --stride 1 --seed 1 --K 3" ""
t "--H 21 --W 17 --N 8 --M 24 --P 8 --stride 2 --seed 3 --K 3" "40 60 56"
t "--H 22 --W 19 --N 16 --M 48 --P 24 --stride 2 --seed 6 --K 3" "40 70 8"
t "--H 7 --W 7 --N 8 --M 48 --P 56 --stride 1 --seed 8 --K 3" ""
# --- 5x5
t "--H 9 --W 9 --N 8 --M 16 --P 8 --stride 1 --seed 2 --K 5" ""
t "--H 10 --W 8 --N 16 --M 24 --P 8 --stride 2 --seed 3 --K 5" ""
t "--H 3 --W 3 --N 8 --M 8 --P 4 --stride 1 --seed 11 --K 5" ""
t "--H 4 --W 6 --N 8 --M 16 --P 8 --stride 1 --seed 12 --K 5" ""
t "--H 14 --W 14 --N 24 --M 48 --P 8 --stride 1 --seed 13 --K 5" ""
t "--H 13 --W 11 --N 8 --M 24 --P 8 --stride 2 --seed 14 --K 5" ""
t "--H 20 --W 14 --N 8 --M 24 --P 8 --stride 1 --seed 15 --K 5" "30 40 56"
t "--H 21 --W 17 --N 16 --M 24 --P 16 --stride 2 --seed 16 --K 5" "40 60 8"
t "--H 12 --W 12 --N 8 --M 24 --P 16 --stride 1 --seed 17 --K 5" "1024 4096 8"
t "--H 7 --W 7 --N 96 --M 48 --P 8 --stride 1 --seed 18 --K 5" ""
t "--H 9 --W 9 --N 8 --M 16 --P 8 --stride 1 --seed 19 --K 5 --pos 0.5" ""
# --- activation tables (hard-swish ready)
t "--H 8 --W 8 --N 8 --M 16 --P 8 --stride 1 --seed 4 --K 3 --lut" ""
t "--H 9 --W 9 --N 8 --M 16 --P 8 --stride 1 --seed 5 --K 5 --lut" ""
t "--H 15 --W 13 --N 16 --M 24 --P 8 --stride 2 --seed 20 --K 5 --lut" "40 60 56"
# --- squeeze-and-excite (two passes: exact squeeze sums + gated projection), 3x3 and 5x5
t "--H 8 --W 8 --N 8 --M 24 --P 8 --stride 2 --seed 30 --K 3 --se" ""
t "--H 9 --W 9 --N 8 --M 16 --P 8 --stride 1 --seed 31 --K 5 --se" ""
t "--H 10 --W 10 --N 16 --M 24 --P 16 --stride 2 --seed 32 --K 5 --se --lut" ""
t "--H 6 --W 6 --N 8 --M 24 --P 16 --stride 1 --seed 33 --K 5 --se" "1024 4096 8"
t "--H 7 --W 7 --N 24 --M 48 --P 12 --stride 1 --seed 34 --K 5 --se --lut" "1024 4096 4"
t "--H 14 --W 14 --N 24 --M 96 --P 32 --stride 2 --seed 35 --K 5 --se --lut" ""
t "--H 7 --W 7 --N 72 --M 432 --P 72 --stride 1 --seed 36 --K 5 --se --lut" ""
[ $fail -eq 0 ] && echo "ALL RTL BLOCK TESTS PASSED" || { echo "SOME RTL BLOCK TESTS FAILED"; exit 1; }
