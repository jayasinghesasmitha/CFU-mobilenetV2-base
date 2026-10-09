# Fused CFU for the full MobileNetV3-small: 5x5 depthwise, hard-swish and squeeze-and-excite

## 0. What the model contains (`model/mobilenetv3_small_full_int8.tflite`, 114 operators)

| | count | where |
|---|---|---|
| depthwise 5x5 | **8** | blocks 3..10 of the network (stride 1 or 2, SAME padding) |
| depthwise 3x3 | 3 | the first block (no expansion conv, with SE) and two plain blocks (E -> DW -> P, ReLU) |
| `HARD_SWISH` | 19 | stem, two per 5x5 block (after the expansion conv and after the depthwise), head |
| squeeze-and-excite | 9 | `MEAN -> CONV(1x1, ReLU) -> CONV(1x1, ReLU6) -> MUL(const) -> MUL(feature x gate)` |
| other | | input rescale (`MUL`,`ADD`), 7 residual `ADD`, global pool `MEAN`, 2 head convs, softmax |

Fusable inverted-residual blocks (`E -> [HS] -> DW -> [HS] -> [SE] -> P`): **10** - 8 with 5x5 + hard-swish + SE and 2 plain
3x3 blocks. The very first block (depthwise + SE + pointwise, no expansion conv) and the stem/head stay on the CPU.
`tools/gen_fused_plan.py` finds them (and lists any block it must leave on the CPU).

## 1. 5x5 depthwise (the main extension)

Input buffer = 9 banks by `(row mod 3, col mod 3)`: every 3x3 window reads in one clock, but a 5x5 window puts up to **4
pixels in one bank**, so with this banking at least 4 reads per bank (4 passes) are unavoidable. The CFU therefore
reads a 5x5 window as **four 3x3 sub-windows** with origins (0,0) (0,3) (3,0) (3,3) (36 slots, 25 real taps; the other 11
carry weight 0).

```
for each output pixel:                 3x3: S = 1 sub-window            5x5: S = 4 sub-windows
  PREP    window origin / bank addresses / validity of every sub-window            (2 clocks each)
  for each expanded channel m:
    for each sub-window s:             <- new loop
      for each 8-channel input chunk:  issue (m,s,chunk): one per clock
         9 expansion engines -> requantise -> [activation table] -> F1 (9 values)
         depthwise: 9 products of sub-window s with its weights -> dsum
    dacc += dsum over s (new accumulator, no partial-sum RAM)
    after the last sub-window: requantise -> [table] -> F2 -> [gate] -> projection
```
* Reused unchanged: 9 banks, 9 expansion engines, padding mask, requantiser, projection engines.
* Depthwise weights: 9 banks addressed `{sub_window, channel>>2}` (one bank per tap slot).
* Kernel size is a per-block config register (cfg 24), so one network can mix 3x3 and 5x5.
* Cost: `cycles = 43 + P + pixels*(2S + S*M*N/8) + (pixels-1)*max(0, P+4 - S*M*N/8)`, S = 1 or 4 (RTL-verified;
  equals the hardware counter in every single-job test). 5x5 = **4x** the cycles of the same 3x3 block for 2.78x the taps
  (25/36 = 69% tap efficiency). Alternatives (25 expansion engines: ~200 DSPs, does not fit; 3 passes: impossible with
  mod-3 banking; an F1 buffer: contradicts the paper's zero-buffer goal) were rejected.

## 2. Hard-swish = three 256-entry activation tables

An int8 -> int8 function depends on 8 bits only. The CFU has a table after each requantiser (expansion, depthwise,
projection), loaded with `MEMW` 21/22/23 and enabled by cfg 25. The firmware fills a table by running TFLM's own
`reference_ops::HardSwish<int8_t>` on the 256 possible inputs (`fused_dsc.cc: OnHardSwish`) - exact by construction.
The depthwise input zero point is taken from the depthwise operator's own parameters (the table output's), and
out-of-image taps contribute 0, which is what TFLite's skipped taps do.

## 3. Squeeze-and-excite (implemented: two passes, no feature map stored)

The gate of every pixel depends on the mean of F2 over the **whole** map, so a pixel-wise pipeline cannot finish a pixel
alone. Solution: compute F2 twice.

```
TFLM op        what happens
-------------  ----------------------------------------------------------------------------------------
E, HS, DW, HS  captured (arguments stashed, nothing computed)
MEAN           PASS 1 (CFU, mode SQUEEZE): E -> DW for every pixel; each F2 value is added into a per-channel
               sum RAM; the CPU reads the M sums (SUMRD) and produces the MEAN output tensor with TFLM's own
               float arithmetic (QuantizedMeanOrSum, or the integer Mean when input/output quantisation match)
CONV, CONV     run on the CPU as usual (tiny: 1x1 spatial)
MUL (const)    runs on the CPU as usual
MUL (gating)   captured: gate vector + TFLite MUL parameters (zero points, multiplier, shift, activation range)
P (1x1 conv)   PASS 2 (CFU, mode EXCITE): E -> DW again, F2 x gate with TFLite int8 MUL arithmetic
               (extra requantiser, 7 clocks), projection, output written into the P output tensor
```
* Weights and the input stay resident in the CFU between the passes (no reload); only the projection weights and the
  gate are loaded for pass 2. P > 56 runs pass 2 once per output group.
* Hardware added: 1024 x 32-bit sum RAM + 2-clock read-modify-write, a gate RAM, one `dsc_requant` stage (+4 DSP),
  mode/gate config registers (cfg file grown to 64), instruction `SUMRD`, tag pipeline 23 -> 31 stages.
* Cost: E and DW are computed twice, so an SE block takes ~2x the cycles of the same block without SE (see the table in
  the README); the price of keeping zero intermediate feature maps.
* Limits: an SE block must fit ONE row strip (input slots <= 1024, output words <= 4096); all 8 SE blocks of this model do.
  Blocks that do not fit are left on the CPU by the plan generator.

## 4. TFLM integration (what is hooked)

`CONV_2D`, `DEPTHWISE_CONV_2D`, `PAD`, `HARD_SWISH`, `MEAN`, `MUL` kernels carry a one-line hook
(`src/tensorflow/.../*.cc|h`). Operators are numbered in graph order (only those six types) by `gen_fused_plan.py`; the
firmware counts the same way (reset by `ResetStats()` before each inference) and shape-checks every captured call.
A mismatch aborts with a clear message instead of producing wrong results.

## 5. Verified (PC) / not verified

Verified: 25 RTL block tests (3x3, 5x5, strips, groups, tables, **7 SE cases incl. a model-sized 7x7x72->432->72 block**;
squeeze sums exact, cycle formula = hardware counter); the full model through real TFLM + real RTL (10/10 blocks,
bit-identical to plain TFLM, per-block C-reference check 0 mismatches); synthetic models with 3x3/5x5, hard-swish, PAD,
stride 1/2 and SE (both MEAN paths, P = 72) x 3 weight sets x 2 inputs, all bit-identical to plain TFLM.
Not verified: RISC-V firmware build, board / Renode run, Vivado resources and timing of this RTL.
