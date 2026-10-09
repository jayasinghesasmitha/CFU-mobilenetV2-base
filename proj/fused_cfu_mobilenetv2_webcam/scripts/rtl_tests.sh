#!/bin/sh
# RTL block tests (PC): the real C driver (dsc_run_block) -> instruction log -> real RTL (Verilator) -> responses
# -> driver reassembles the output -> compared with the Python golden model.  Covers stride 1/2, odd sizes,
# row-strip tiling and projection channel groups (limits are shrunk on purpose to force the tiling paths).
# Needs: verilator, gcc, python3 + numpy.
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
      echo "$out" | grep -q "PASS" || fail=1; }
t "--H 8 --W 8 --N 8 --M 16 --P 8 --stride 1 --seed 1" ""
t "--H 20 --W 14 --N 8 --M 24 --P 8 --stride 1 --seed 2" "30 40 56"
t "--H 21 --W 17 --N 8 --M 24 --P 8 --stride 2 --seed 3" "40 60 56"
t "--H 20 --W 14 --N 16 --M 32 --P 8 --stride 1 --seed 4" "45 140 56"
t "--H 12 --W 12 --N 8 --M 24 --P 16 --stride 1 --seed 5" "1024 4096 8"
t "--H 22 --W 19 --N 16 --M 48 --P 24 --stride 2 --seed 6" "40 70 8"
t "--H 10 --W 10 --N 8 --M 24 --P 12 --stride 1 --seed 7" "1024 4096 4"
[ $fail -eq 0 ] && echo "ALL RTL BLOCK TESTS PASSED" || { echo "SOME RTL BLOCK TESTS FAILED"; exit 1; }
