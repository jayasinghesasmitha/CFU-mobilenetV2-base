# MobileNetV2 CPU Baseline — CFU-Playground Option A

This is the CPU-only project overlay. It is designed to live beside the
upstream `mnv2_first` project:

    CFU-Playground/
    └── proj/
        ├── mnv2_first/
        └── mnv2_baseline/

## Model

The binary model is intentionally NOT included. Put your model in:

    mnv2_baseline/model/model_mobilenetv2_160_035.tflite

The CFU-Playground build will convert the `.tflite` model to a C header.

## Inputs

Put images under:

    inputs/

The helper:

    python3 tools/jpg_to_dat.py inputs/cat.jpg inputs/cat.dat

creates resized RGB bytes. For a final int8 input tensor, use the model's
input scale and zero-point; do not assume a generic JPEG byte mapping.

## Build / Renode

From the project directory:

    cd ~/CFU-Playground/proj/mnv2_baseline
    make renode

For a clean rebuild:

    make clean
    make renode NO_CFU=1 SW_ONLY=1

The baseline sets `SW_ONLY=1`, so the Renode run does not build a Verilator CFU.

## Cycle measurement

The standard CFU-Playground MobileNetV2 application measures the complete
TFLite Micro `Invoke()` using the RISC-V cycle counter. Use the printed total
cycle count for the CPU-vs-CFU comparison.

## Important

This package contains the project-specific files and helper tools. It does
not contain the CFU-Playground framework (`common/`, `soc/`, `third_party/`,
`proj.mk`, Renode infrastructure, TFLite Micro, etc.).
