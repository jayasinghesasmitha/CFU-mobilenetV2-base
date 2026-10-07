# fused_cfu_with_mobilenetv2

MobileNetV2 a0.35 INT8 running on the VexRiscv/LiteX/CFU-Playground stack with the **fused
Expansion -> Depthwise -> Projection CFU** (Yildirim & Ozturk, "RISC-V Based TinyML Accelerator for Depthwise
Separable Convolutions in Edge AI") instead of the existing `mnv2_first` 1x1 CFU.

It is the counterpart of `proj/mnv2_cfu_package`: **same model, same embedded cat image, same application and
output format** - only the CFU (and the TFLM kernel hooks that feed it) differ, so the two CFU implementations can be
compared in the same environment.

| project | CFU | what it accelerates |
|---|---|---|
| `mnv2_baseline` | none | nothing (CPU only) |
| `mnv2_cfu_package` | existing `mnv2_first` | eligible 1x1 CONV_2D |
| **`fused_cfu_with_mobilenetv2`** | **fused DSC CFU (this project)** | **all 16 inverted-residual blocks, whole Ex->Dw->Pr** |

## Layout

```
fused_cfu_with_mobilenetv2/
  Makefile                      same structure as mnv2_cfu_package (MNv2_BASELINE, include ../proj.mk)
  cfu.v                         the fused CFU (single Verilog file, module Cfu)
  model/mobilenetv2_a035_224_int8.tflite
  inputs/cat.jpg                tools/image_to_header.py regenerates src/cat_image.h
  src/
    mnv2_app.cc                 application (same output as the other projects + fused statistics block)
    cat_image.h / .dat          embedded image (identical to the other projects)
    dsc_driver.c/.h             CFU driver: loads weights / feature maps, runs, reads results
                                (row-strip tiling for big maps, channel groups for P > 56)
    dsc_ref.c                   layer-by-layer C reference (used by FUSED_VERIFY)
    fused_dsc.h/.cc             TFLM integration: captures E / PAD / D, runs the whole block at P
    fused_plan.h                GENERATED: which operators form a block (tools/gen_fused_plan.py)
    software_cfu.cc             template stub (hardware CFU is used)
    tensorflow/lite/kernels/internal/reference/integer_ops/{conv.h,conv.cc,depthwise_conv.h}
    tensorflow/lite/micro/kernels/pad.cc                         patched kernels with the fused hooks
  tools/  gen_fused_plan.py  compare_runs.py  gen_vectors.py  image_to_header.py
  sim/    host_mnv2.cpp  Makefile.host  tb_replay.v  block_test.c  expected_host_output.txt
  scripts/ run_host_test.sh  rtl_tests.sh  block_test.sh
  docs/   CFU_INSTRUCTIONS_AND_PIPELINE.md
```

## How the fusion works inside TFLM

TFLM runs `CONV_2D(1x1) -> [PAD] -> DEPTHWISE_CONV_2D(3x3) -> CONV_2D(1x1)` one operator at a time. The kernels were
patched so the unchanged graph behaves like this:

1. **Expansion conv**: arguments are stashed, nothing is computed (F1 is never written).
2. **PAD** (stride-2 blocks only): skipped. Out-of-image taps are masked in hardware (the value TFLite pads with equals
   the F1 zero point, so the contribution is exactly zero).
3. **Depthwise conv**: arguments stashed, nothing computed (F2 is never written).
4. **Projection conv**: the whole block runs on the CFU and the result goes straight into the projection output tensor,
   so the following residual `ADD` is untouched.

Blocks are located by operator index (`src/fused_plan.h`, generated from the model) and every stashed call is
shape-checked; a mismatch aborts with a clear message instead of giving wrong results.
The first inverted-residual block of the network (expansion factor 1, no 1x1 expansion), the stem conv, the final
1x1 conv, pooling and the FC layer stay on the CPU - so a 1x1-only CFU still accelerates some operators this one does not.
That is part of what the comparison measures.

## Install

```
CFU-mobilenetV2-base/proj/
    mnv2_baseline/            (existing)
    mnv2_cfu_package/         (existing)
    fused_cfu_with_mobilenetv2/    <- put this folder here
```

No vendoring script is needed (cfu.v is hand-written). To regenerate the plan for a different model:
`python3 tools/gen_fused_plan.py model/<model>.tflite src/fused_plan.h` (needs `pip install tflite`).

## Build and run

Real hardware (Nexys A7 / Nexys 4 DDR, same commands as your other projects):

```
conda activate cfu-common
cd proj/fused_cfu_with_mobilenetv2
make clean
make prog TARGET=digilent_nexys4ddr USE_VIVADO=1 UART_SPEED=115200      # long: Vivado
make load TARGET=digilent_nexys4ddr USE_VIVADO=1 UART_SPEED=115200      # prints the same report as mnv2_cfu_package
```

Optional cross-check of every fused block against the layer-by-layer C reference (slower, big static buffers):

```
make load ... FUSED_VERIFY=1       # the report ends with "Verification vs C reference: 16 blocks checked, 0 mismatching bytes -> PASS"
```

Renode (`make renode`) works for correctness only: Renode does not count CFU time, so its cycle numbers are not
meaningful for either CFU. Use real hardware for the comparison.

## Comparing the two CFUs

1. Run `mnv2_cfu_package` and `fused_cfu_with_mobilenetv2` on the SAME board/target, same image, and save each console output
   to a text file (optionally also the CPU-only `mnv2_baseline`).
2. `python3 tools/compare_runs.py existing_cfu.log fused_cfu.log [cpu_only.log]`

It checks image checksum, top-1, output statistics and the first 20 FLOAT32 outputs (they must be identical), then prints
the whole-model cycle counts and speedups. `sim/expected_host_output.txt` holds the reference outputs for the cat image.

The fused run additionally prints a per-block table: CFU busy cycles (hardware counter) and CPU cycles (weights + input
load + compute + read-back) for each of the 16 blocks.

## Tests (PC, no board needed)

```
scripts/rtl_tests.sh        # real driver + real RTL (Verilator): 7 tiling / stride / group cases vs golden model
scripts/run_host_test.sh    # real TFLM + patched kernels + real RTL: whole MobileNetV2, plain vs fused, bit-exact
VERIFY=1 scripts/run_host_test.sh
```
They need g++, verilator (>= 5), make, python3 + numpy, and the project placed inside the CFU-Playground checkout
(they use `../../third_party/tflite-micro`).

## What has and has not been verified

Verified on a PC (this package's tests):
* RTL block tests pass (stride 1/2, odd sizes, forced row strips, forced channel groups).
* Whole MobileNetV2 through real TFLM with the real RTL: output of the fused run is bit-identical to plain TFLM
  (1000 FLOAT32 values, top-1 = 64, FNV1a 0x2d8f8f61); all 16 blocks run on the CFU (19 jobs, 18 strips);
  `FUSED_VERIFY` reports 0 mismatching bytes. CFU busy cycles for the 16 blocks: 1,837,367.
* CFU instruction mix for one inference: 135,610 MEMW, 34,790 OUTRD, 407 CFG (+ STATUS polls that overlap compute).

NOT verified (needs you):
* The RISC-V firmware build and a run on the board or in Renode (only syntax-checked here).
* Vivado resource/timing for this exact build (earlier standalone estimate of the CFU alone: ~38k LUT, ~196 DSP, ~70 BRAM36;
  the SoC adds to that). If it does not fit, lower `NPE` in `cfu.v` (module `Cfu`) - blocks need up to 56 output channels per pass,
  so smaller values simply mean more passes for the 56/112-channel blocks.
* Real cycle counts and therefore which CFU is faster overall.

## Limits

* Per pass: P <= 56 (larger P is split into groups), N and M multiples of 8, P multiple of 4, M*N <= 32768, H/W <= 255.
* Input and output buffers are on-chip; larger maps are processed in row strips (extra halo rows are re-loaded).
* Weights are re-loaded for every block on every inference (the CFU memories are shared by all blocks); this cost is part
  of the measured total.
* Other models: regenerate `src/fused_plan.h`; unsupported blocks must be left out of the plan (they then run on the CPU).
