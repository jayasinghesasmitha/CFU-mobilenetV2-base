#!/bin/sh
# usage: scripts/block_test.sh "<gen_vectors args>" [slots words group]
# Runs the real driver (dsc_run_block) against the RTL (Verilator) and checks the result vs the golden model.
set -e
D=build_tb/$(echo "$1" | tr -c 'A-Za-z0-9' '_')
mkdir -p $D
python3 tools/gen_vectors.py --outdir $D $1 | head -1
gcc -O1 -DDSC_CFU_STUB -DDSC_WAIT_BLOCKING -Isrc -I$D -o $D/block_test sim/block_test.c src/dsc_driver.c
./$D/block_test log $D $2 $3 $4
./build_tb/obj_dir/simv +dir=$D | grep -E "replayed|TIMEOUT"
./$D/block_test check $D $2 $3 $4
