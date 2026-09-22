# Repository changes

## Added project

All implementation changes are isolated under:

`proj/mnv2_cfu_depthwise/`

Files:

- `Makefile` — CFU-Playground project integration and host-test targets.
- `entrypoint.sh` — reproducible test/generate/image/build/clean entrypoint.
- `cfu_gen.py` — Amaranth-to-Verilog generator.
- `gateware/fused_depthwise.py` — synthesizable fused CFU command engine.
- `gateware/test_fused_depthwise.py` — protocol and arithmetic simulation tests.
- `src/mnv2_cfu.h` — software-visible custom instruction ABI.
- `src/fused_depthwise.h`, `src/fused_depthwise.cc` — C++ reference and CFU driver.
- `src/software_cfu.cc` — software emulation of the CFU ABI.
- `src/mnv2_app.cc` — firmware demonstration reading the compiled custom image.
- `src/input_image.h` — replaceable compiled image array (black placeholder by default).
- `inputs/README.md`, `tools/image_to_header.py` — exact custom-image conversion workflow.
- `tests/test_image_conversion.py` — verifies RGB order, bilinear resize, and 150,528-byte output.
- `tests/fused_image_model.py` — full-image zero-intermediate-buffer model with on-the-fly padding.
- `tests/test_image_fusion.py` — padding, stride, signed data, saturation and shape tests.
- `tests/test_reference.py` — randomized reference checks.
- `tests/test_no_dma.py` — static verification of the instruction-only data path.
- `tests/test_cxx_reference.cc`, `tests/test_software_cfu.cc` — host C++ checks.
- `README.md`, `PAPER_MAPPING.md` — usage, architecture, traceability, and limitations.

## Shared files

No shared repository source file was modified. Existing projects, including
`proj/mnv2_cfu_package`, therefore retain their original sources and behavior.
The only repository-wide filesystem operation performed during validation was
initializing the already-declared Git submodules.
