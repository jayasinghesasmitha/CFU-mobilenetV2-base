# mobilenetv3_full_cfu_fused

The **full MobileNetV3-small** (`model/mobilenetv3_small_full_int8.tflite`: 8 depthwise **5x5** layers, 19 **hard-swish**,
9 **squeeze-and-excite** modules) on VexRiscv/LiteX/CFU-Playground with the **fused Expansion -> Depthwise -> Projection CFU**,
extended for 5x5 depthwise, hard-swish and squeeze-and-excite. Same layout, embedded image and output format as the other
fused projects, so `tools/compare_runs.py` can compare any of them.

10 inverted-residual blocks run on the CFU: **8 blocks with 5x5 depthwise + hard-swish + squeeze-and-excite** and 2 plain 3x3 blocks.
Stem, the first block (no expansion conv), residual adds, input rescale, global pooling, the SE mini-convs and the head stay on the CPU.

## What the CFU does that the MobileNetV2 CFU did not

| feature | how (details: docs/DESIGN_5x5_HSWISH_SE.md) |
|---|---|
| depthwise 3x3 **or 5x5**, per block | a 5x5 window is read as four 3x3 sub-windows on the same 9 banks / 9 engines; partial sums added in a new accumulator; cfg register 24 selects K |
| hard-swish | three 256-entry int8 tables (after expansion / depthwise / projection), filled with TFLM's own function: exact |
| squeeze-and-excite | two passes with no feature map stored: pass 1 sums F2 per channel (the CPU turns the sums into the exact `MEAN` output), pass 2 recomputes F2, multiplies by the gate (TFLite int8 `MUL` arithmetic) and projects |
| bigger buffers | expansion weights up to 65,536 bytes (96 -> 576 blocks), output groups for P > 56, row strips for big maps |

## Layout
```
mobilenetv3_full_cfu_fused/
  Makefile  cfu.v  model/  inputs/        same structure as the other fused projects
  src/    mnv3_app.cc  dsc_driver.c/.h  dsc_ref.c  fused_dsc.h/.cc  fused_plan.h (GENERATED)  cat_image.h  software_cfu.cc
          tensorflow/.../integer_ops/{conv.h,conv.cc,depthwise_conv.h}
          tensorflow/lite/micro/kernels/{pad,hard_swish,reduce_common,mul}.cc        <- hooked TFLM kernels
  tools/  gen_fused_plan.py  make_test_model.py  gen_vectors.py  compare_runs.py  image_to_header.py
  sim/    host_mnv2.cpp  Makefile.host  tb_replay.v  block_test.c  test_models/  expected_host_output.txt
  scripts/ run_host_test.sh  rtl_tests.sh  block_test.sh
  docs/   DESIGN_5x5_HSWISH_SE.md
```

## Build and run (board)
```
CFU-mobilenetV3-base/proj/mobilenetv3_full_cfu_fused/        <- place the folder next to the other projects
conda activate cfu-common
cd proj/mobilenetv3_full_cfu_fused
make clean
make prog TARGET=digilent_nexys4ddr USE_VIVADO=1 UART_SPEED=115200      # long (Vivado)
make load TARGET=digilent_nexys4ddr USE_VIVADO=1 UART_SPEED=115200      # prints the report
```
`make load ... FUSED_VERIFY=1` also cross-checks every fused block against a C reference (slower, big static buffers).
Renode (`make renode`) is for correctness only (it does not count CFU time).
Different model: `python3 tools/gen_fused_plan.py model/<model>.tflite src/fused_plan.h` (it prints which blocks fuse and why
others do not) and update the model name and the output scale/zero point in `src/mnv3_app.cc`.
Compare with another project: `python3 tools/compare_runs.py other.log mobilenetv3_full_cfu_fused.log`.
Reference output for the embedded image: `sim/expected_host_output.txt` (top-1 class 38, score 93750 x10^-6; what plain
TFLM gives - the model's accuracy/preprocessing itself was not evaluated).

## Tests (PC, no board)
```
scripts/rtl_tests.sh        # real driver + real RTL (Verilator): 25 cases (3x3/5x5/strips/groups/tables/squeeze-and-excite)
scripts/run_host_test.sh    # real TFLM + hooked kernels + real RTL: the full model + synthetic 3x3/5x5/HS/PAD/SE models
VERIFY=1 scripts/run_host_test.sh
```
Needs g++, verilator >= 5, make, python3 + numpy, `pip install tflite flatbuffers`, and the project inside the CFU-Playground
checkout (uses `../../third_party/tflite-micro`).

## Results of the PC tests
Full model, plain TFLM vs fused CFU: **identical 1000 outputs**; per-block C-reference check 0 mismatching bytes.
CFU busy cycles (hardware counter, compute only; SE blocks include both passes):

| block | K | SE | input -> output | hw cycles |
|---|---|---|---|---|
| 0 | 3 | - | 56x56x16 -> 28x28x24 | 114,598 |
| 1 | 3 | - | 28x28x24 -> 28x28x24 | 208,678 |
| 2 | 5 | SE | 28x28x24 -> 14x14x32 | 454,858 |
| 3 | 5 | SE | 14x14x32 -> 14x14x32 | 1,207,498 |
| 4 | 5 | SE | 14x14x32 -> 14x14x32 | 1,207,498 |
| 5 | 5 | SE | 14x14x32 -> 14x14x40 | 605,394 |
| 6 | 5 | SE | 14x14x40 -> 14x14x40 | 944,082 |
| 7 | 5 | SE | 14x14x40 -> 7x7x72 | 706,997 |
| 8 | 5 | SE | 7x7x72 -> 7x7x72 | 2,287,541 |
| 9 | 5 | SE | 7x7x72 -> 7x7x72 | 2,287,541 |
| total | | | | **10,024,685** (23 jobs) |

CFU instruction mix for one inference: 119,652 MEMW, 20,678 OUTRD, 423 CFG, ~1,800 SUMRD (+ STATUS polls that overlap compute).
Synthetic models (3x3/5x5, hard-swish, explicit PAD, stride 1/2, SE with both MEAN quantisation paths, P = 72): 7/7 blocks fused,
bit-identical to plain TFLM for 3 weight sets x 2 inputs.

## Resources (rough Yosys xc7 estimate of the CFU alone - NOT a Vivado result)
~41k LUT, ~8.8k FF, 202 DSP48, ~85 BRAM36-equivalents (before the VexRiscv/LiteX SoC). The Nexys A7-100T has 63,400 LUT,
240 DSP, 135 BRAM36, so it will be tight. If Vivado does not fit it, lower `NPE` (projection engines) in `cfu.v`: blocks with
P > NPE then run in more passes.

## NOT verified (needs you)
* RISC-V firmware build and a run on the board / in Renode (the firmware-side sources were only syntax-checked), so no real board cycle counts.
* Vivado resources and timing of this exact RTL.
* The model's classification accuracy (the output is bit-identical to plain TFLM, which is what a CFU must reproduce).
