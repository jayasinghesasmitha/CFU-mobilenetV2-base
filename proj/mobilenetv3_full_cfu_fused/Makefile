#!/bin/env python

# MobileNetV3-small a0.75 INT8 + FUSED DSC CFU project (3x3 and 5x5 depthwise).
#
# Same embedded cat image and output format as the MobileNetV2 projects, with the MobileNetV3-small model.
# The CFU in cfu.v is the fused Expansion -> Depthwise(3x3|5x5) -> Projection accelerator
# (Yildirim & Ozturk, extended for 5x5 depthwise and activation tables).
#
#   cfu.v                                  hand-written Verilog CFU (no cfu_gen.py needed)
#   src/tensorflow/.../integer_ops/*       conv + depthwise kernels with the fused-CFU hooks
#   src/tensorflow/lite/micro/kernels/{pad,hard_swish}.cc   PAD / HARD_SWISH folded into the fused block
#   src/fused_plan.h                       blocks to fuse (python3 tools/gen_fused_plan.py ...)

export DEFINES :=
DEFINES += NDEBUG

# Enable the fused-CFU kernel hooks (see src/fused_dsc.cc).
DEFINES += FUSED_DSC

# Optional: cross-check every fused block against the layer-by-layer C reference
# (slow, uses the otherwise-unused F1/F2 tensors).   make ... FUSED_VERIFY=1
ifdef FUSED_VERIFY
DEFINES += FUSED_VERIFY
endif

# Keep the model and application isolated from upstream mnv2_first.
export MNv2_BASELINE := 1

# The model is an external project model copied from model/.
DEFINES += INCLUDE_MODEL_PDTI8

ifneq 'hps' '$(PLATFORM)'
DEFINES += INCLUDE_MODEL_MNV2
endif

RUN_MENU_ITEMS := 3 1

include ../proj.mk
