#!/usr/bin/env make

# ============================================================
# MobileNetV2 Depthwise CFU Project
# ============================================================
#
# This project uses:
#
#   SAME MobileNetV2 model as mnv2_baseline
#   SAME TFLite Micro inference framework
#   SAME input image
#
# The difference is that this project also contains the
# fused Expansion -> Depthwise -> Projection CFU.
#
# ============================================================


# ------------------------------------------------------------
# Basic build configuration
# ------------------------------------------------------------

export DEFINES := NDEBUG


# ------------------------------------------------------------
# IMPORTANT:
#
# DO NOT set SKIP_TFLM.
#
# The baseline MobileNetV2 application uses TFLite Micro.
# We need the same model/inference environment here.
# ------------------------------------------------------------


# Tell proj.mk to use the MobileNetV2 project-model handling.
#
# This does two things:
#
#   1. Copies this project's .tflite model into:
#        build/src/models/mnv2/
#
#   2. Removes the upstream generic MobileNetV2 application.
#
export MNv2_BASELINE := 1


# Enable the MobileNetV2 model.
DEFINES += INCLUDE_MODEL_MNV2


# The project menu is:
#
#   3 -> Project
#   1 -> MobileNetV2
#
RUN_MENU_ITEMS := 3 1
TEST_MENU_ITEMS := 3 1


# ------------------------------------------------------------
# Use common project build system
# ------------------------------------------------------------

include ../proj.mk


# ------------------------------------------------------------
# Project targets
# ------------------------------------------------------------

.PHONY: test
.PHONY: test-python
.PHONY: test-cxx
.PHONY: generate
.PHONY: image
.PHONY: clean


# ------------------------------------------------------------
# Input image
# ------------------------------------------------------------

IMAGE ?= inputs/input.jpg


# ------------------------------------------------------------
# Generate input_image.h
# ------------------------------------------------------------

image:
	cd $(CURDIR) && \
	$(PYRUN) tools/image_to_header.py \
		$(IMAGE) \
		src/input_image.h


# ------------------------------------------------------------
# Tests
# ------------------------------------------------------------

test: test-python test-cxx


test-python:
	cd $(CURDIR) && \
	$(PYRUN) -m unittest discover \
		-s gateware \
		-p 'test_*.py'

	cd $(CURDIR) && \
	$(PYRUN) -m unittest discover \
		-s tests \
		-p 'test_*.py'


# ------------------------------------------------------------
# C++ CFU tests
# ------------------------------------------------------------

test-cxx:
	mkdir -p build/tests

	g++ -std=c++11 \
		-Wall \
		-Wextra \
		-Werror \
		-DCFU_SOFTWARE_DEFINED \
		-I$(CFU_ROOT)/common/src \
		-Isrc \
		src/fused_depthwise.cc \
		src/software_cfu.cc \
		tests/test_cxx_reference.cc \
		-o build/tests/test_cxx_reference

	build/tests/test_cxx_reference

	g++ -std=c++11 \
		-Wall \
		-Wextra \
		-Werror \
		-DCFU_SOFTWARE_DEFINED \
		-I$(CFU_ROOT)/common/src \
		-Isrc \
		src/fused_depthwise.cc \
		src/software_cfu.cc \
		tests/test_software_cfu.cc \
		-o build/tests/test_software_cfu

	build/tests/test_software_cfu


# ------------------------------------------------------------
# Generate CFU Verilog
# ------------------------------------------------------------

generate:
	cd $(CURDIR) && \
	$(PYRUN) cfu_gen.py


# ------------------------------------------------------------
# Clean
# ------------------------------------------------------------

clean:
	$(MAKE) -C ../.. clean
	rm -rf build