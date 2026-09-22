# MobileNetV2-inspired fused depthwise CFU

This project is a self-contained experiment derived from `proj/mnv2_cfu_package`.
It implements a paper-inspired **Expansion -> 3x3 Depthwise -> Projection**
operator as an Amaranth CFU, plus a software emulation and a matching C++
reference implementation.

## Scope and honest limitations

The accelerator computes **one output pixel per invocation**. The CPU explicitly
pushes the 3x3 input patch and all weights/biases through the CFU command
protocol. There is **no DMA**, no system-memory port, no tensor address
translation, and no automatic TFLite graph rewrite. Consequently this project
does **not** claim transparent full-model MobileNetV2 fusion. Integrating the
operator into a full model would require a separate TFLite kernel adapter,
layout/quantization policy, tiling strategy, and a benchmark on the target SoC.

The implementation is deliberately bounded to make synthesis and verification
tractable:

* input channels: 1..8;
* expanded channels: 1..8;
* output channels: 1..8;
* nine signed int8 taps per channel;
* signed int32 bias and accumulation;
* each stage uses an arithmetic right shift followed by int8 saturation;
* one result byte is read with each `POP_OUTPUT` command.

This is a reference architecture, not a claim of bit-exact compatibility with
every TensorFlow Lite quantization variant. Per-channel zero points, asymmetric
weights, residual addition, activation functions beyond saturation, and model
packing are intentionally outside this first version.

## Where to put a custom image

The image is **compiled into the firmware**. There is no runtime JPEG reader,
filesystem loader, serial image transfer, or DMA path.

1. Put a JPEG or PNG anywhere convenient; the recommended path is:

   `proj/mnv2_cfu_depthwise/inputs/my_image.jpg`

2. From the repository root, convert it to the compiled C array:

   ```bash
   scripts/pyrun proj/mnv2_cfu_depthwise/tools/image_to_header.py \
     proj/mnv2_cfu_depthwise/inputs/my_image.jpg \
     proj/mnv2_cfu_depthwise/src/input_image.h
   ```

3. Rebuild the firmware/bitstream. `src/mnv2_app.cc` includes
   `src/input_image.h` and reads `input_image[]`.

The actual `mobilenetv2_a035_224_int8.tflite` input metadata is:

* tensor name: `serving_default_input_1:0`;
* shape/layout: `UINT8 [1,224,224,3]` (NHWC);
* channel order: interleaved RGB, not BGR;
* scale: `0.007843137718737125`;
* zero point: `127`.

The converter calls Pillow `convert("RGB")`, then performs a direct 224x224
bilinear resize. It does **not** preserve aspect ratio and does **not** crop. It
writes the resulting raw 0..255 RGB bytes unchanged. In the existing full-model
application those bytes are copied to the tensor by
`common/src/tflite.cc::tflite_set_input_unsigned()`; no host-side mean division
or `[-1,1]` floating-point normalization is applied.

Important: this project remains a standalone fused-CFU reference, not a complete
MobileNetV2 classifier. Its menu demonstration reads the center 3x3 **red
channel** from `input_image[]`, centers bytes using the model zero point
(`value - 127`, saturated to int8), and sends that patch through the fused CFU.
For full-image MobileNetV2 classification today, apply the same conversion to
`proj/mnv2_cfu_package/src/cat_image.h` and use the existing
`mnv2_cfu_package` application; the depthwise project does not yet fuse the
three TFLite graph operators transparently.

## Explicit command protocol

All commands use CFU function 0. The command is encoded in `funct7`; operand 0
contains the payload. `STATUS` returns these bits: bit 0 configured, bit 1 busy,
bit 2 result valid, bit 3 sticky error, bits 8..15 output pop count, and bits
16..23 configured output channels.

| Command | `funct7` | Operand 0 |
| --- | ---: | --- |
| `RESET` | 0 | ignored |
| `CONFIG` | 1 | `Cin[7:0]`, `Cexp[15:8]`, `Cout[23:16]` |
| `SET_SHIFTS` | 2 | expansion, depthwise, projection shifts at bits 0, 8, 16 |
| `PUSH_INPUT` | 3 | signed int8, ordered `[tap][input_channel]` |
| `PUSH_EXP_WEIGHT` | 4 | signed int8, ordered `[expanded][input_channel]` |
| `PUSH_EXP_BIAS` | 5 | signed int32, ordered `[expanded]` |
| `PUSH_DW_WEIGHT` | 6 | signed int8, ordered `[tap][expanded]` |
| `PUSH_DW_BIAS` | 7 | signed int32, ordered `[expanded]` |
| `PUSH_PROJ_WEIGHT` | 8 | signed int8, ordered `[output][expanded]` |
| `PUSH_PROJ_BIAS` | 9 | signed int32, ordered `[output]` |
| `RUN` | 10 | ignored; requires every stream complete |
| `POP_OUTPUT` | 11 | ignored; returns one sign-extended int8 |
| `STATUS` | 12 | ignored |

`RESET` clears all state. `CONFIG` also clears streamed data. A malformed
configuration, overrun stream, incomplete `RUN`, or invalid pop sets the sticky
error bit until reset.

## Files

* `gateware/fused_depthwise.py`: synthesizable Amaranth state machine and
  command protocol.
* `gateware/test_fused_depthwise.py`: focused protocol and fused arithmetic
  simulation tests.
* `src/software_cfu.cc`: software CFU emulation with the same state machine
  contract and reference arithmetic.
* `src/fused_depthwise.cc/.h`: C++ reference fused kernel and explicit CFU
  command-stream API.
* `tests/`: standalone Python and C++ correctness tests.
* `cfu_gen.py`: Amaranth-to-Verilog generator.
* `entrypoint.sh`: `test`, `generate`, `build`, and `clean` convenience commands.

## Tests

From the repository root:

```sh
cd proj/mnv2_cfu_depthwise
./entrypoint.sh test
```

The Python tests require the repository's Amaranth/`amaranth_cfu` environment.
The C++ tests use only the local reference/emulation sources and common CFU
headers. Verilog generation is available with `./entrypoint.sh generate` when
Amaranth and its Verilog backend are installed.

## Build integration

The Makefile includes the normal CFU-Playground `proj.mk` integration and keeps
TFLite disabled (`SKIP_TFLM=1`) because this project intentionally exposes a
focused fused primitive instead of pretending to accelerate an entire model.
Use `make generate` to create `cfu.v`; use `make test` for the self-contained
correctness suite.
