#include "fused_depthwise.h"

#include <stddef.h>
#include <stdint.h>

#include "mnv2_cfu.h"


namespace {


// ============================================================
// Parameter validation
// ============================================================

bool Valid(
    const Mnv2FusedParams& p,
    const int8_t* input,
    int8_t* output) {

  if (input == nullptr || output == nullptr) {
    return false;
  }

  if (p.expansion_weights == nullptr ||
      p.expansion_bias == nullptr ||
      p.depthwise_weights == nullptr ||
      p.depthwise_bias == nullptr ||
      p.projection_weights == nullptr ||
      p.projection_bias == nullptr) {

    return false;
  }


  if (p.input_channels <= 0 ||
      p.input_channels > MNV2_FUSED_MAX_INPUT_CHANNELS) {

    return false;
  }


  if (p.expanded_channels <= 0 ||
      p.expanded_channels > MNV2_FUSED_MAX_EXPANDED_CHANNELS) {

    return false;
  }


  if (p.output_channels <= 0 ||
      p.output_channels > MNV2_FUSED_MAX_OUTPUT_CHANNELS) {

    return false;
  }


  if (p.expansion_shift < 0 ||
      p.expansion_shift > 31) {

    return false;
  }


  if (p.depthwise_shift < 0 ||
      p.depthwise_shift > 31) {

    return false;
  }


  if (p.projection_shift < 0 ||
      p.projection_shift > 31) {

    return false;
  }


  return true;
}


// ============================================================
// Arithmetic shift + INT8 saturation
// ============================================================

int8_t ShiftAndClamp(
    int32_t value,
    int shift) {

  const int32_t shifted =
      value >> shift;


  if (shifted > 127) {
    return 127;
  }


  if (shifted < -128) {
    return -128;
  }


  return static_cast<int8_t>(shifted);
}

}  // namespace


// ============================================================
// Software reference implementation
// ============================================================

bool Mnv2FusedReference(
    const Mnv2FusedParams& p,
    const int8_t* input_patch,
    int8_t* output) {

  if (!Valid(p, input_patch, output)) {
    return false;
  }


  int8_t expanded[
      MNV2_FUSED_KERNEL_TAPS]
      [MNV2_FUSED_MAX_EXPANDED_CHANNELS] = {};


  int8_t depthwise[
      MNV2_FUSED_MAX_EXPANDED_CHANNELS] = {};


  // ----------------------------------------------------------
  // Expansion 1x1
  // ----------------------------------------------------------

  for (int tap = 0;
       tap < MNV2_FUSED_KERNEL_TAPS;
       ++tap) {

    for (int e = 0;
         e < p.expanded_channels;
         ++e) {

      int32_t acc =
          p.expansion_bias[e];


      for (int i = 0;
           i < p.input_channels;
           ++i) {

        acc +=
            static_cast<int32_t>(
                input_patch[
                    tap * p.input_channels + i])
            *
            static_cast<int32_t>(
                p.expansion_weights[
                    e * p.input_channels + i]);
      }


      expanded[tap][e] =
          ShiftAndClamp(
              acc,
              p.expansion_shift);
    }
  }


  // ----------------------------------------------------------
  // 3x3 Depthwise
  // ----------------------------------------------------------

  for (int e = 0;
       e < p.expanded_channels;
       ++e) {

    int32_t acc =
        p.depthwise_bias[e];


    for (int tap = 0;
         tap < MNV2_FUSED_KERNEL_TAPS;
         ++tap) {

      acc +=
          static_cast<int32_t>(
              expanded[tap][e])
          *
          static_cast<int32_t>(
              p.depthwise_weights[
                  tap * p.expanded_channels + e]);
    }


    depthwise[e] =
        ShiftAndClamp(
            acc,
            p.depthwise_shift);
  }


  // ----------------------------------------------------------
  // Projection 1x1
  // ----------------------------------------------------------

  for (int o = 0;
       o < p.output_channels;
       ++o) {

    int32_t acc =
        p.projection_bias[o];


    for (int e = 0;
         e < p.expanded_channels;
         ++e) {

      acc +=
          static_cast<int32_t>(
              depthwise[e])
          *
          static_cast<int32_t>(
              p.projection_weights[
                  o * p.expanded_channels + e]);
    }


    output[o] =
        ShiftAndClamp(
            acc,
            p.projection_shift);
  }


  return true;
}


// ============================================================
// Run complete image
// ============================================================

bool Mnv2FusedRunImage(
    const Mnv2FusedParams& p,
    const int8_t* input,

    int input_height,
    int input_width,

    int stride,
    int padding,

    int input_zero_point,

    int8_t* output,
    int output_capacity,

    int* output_height,
    int* output_width) {

  if (!input ||
      !output ||
      !output_height ||
      !output_width) {

    return false;
  }


  if (input_height <= 0 ||
      input_width <= 0) {

    return false;
  }


  if (stride != 1 &&
      stride != 2) {

    return false;
  }


  if (padding != 0 &&
      padding != 1) {

    return false;
  }


  if (input_zero_point < -128 ||
      input_zero_point > 127) {

    return false;
  }


  if (!Valid(p, input, output)) {
    return false;
  }


  const int oh =
      (input_height + 2 * padding - 3)
      / stride + 1;


  const int ow =
      (input_width + 2 * padding - 3)
      / stride + 1;


  if (oh <= 0 ||
      ow <= 0) {

    return false;
  }


  const int required =
      oh * ow * p.output_channels;


  if (output_capacity < required) {
    return false;
  }


  *output_height = oh;

  *output_width = ow;


  int8_t patch[
      MNV2_FUSED_KERNEL_TAPS *
      MNV2_FUSED_MAX_INPUT_CHANNELS];


  // ----------------------------------------------------------
  // Output pixels
  // ----------------------------------------------------------

  for (int oy = 0;
       oy < oh;
       ++oy) {

    for (int ox = 0;
         ox < ow;
         ++ox) {


      // ------------------------------------------------------
      // Construct 3x3 patch
      // ------------------------------------------------------

      for (int ky = 0;
           ky < 3;
           ++ky) {

        const int iy =
            oy * stride +
            ky -
            padding;


        for (int kx = 0;
             kx < 3;
             ++kx) {

          const int ix =
              ox * stride +
              kx -
              padding;


          const int tap =
              ky * 3 + kx;


          for (int c = 0;
               c < p.input_channels;
               ++c) {

            if (iy >= 0 &&
                iy < input_height &&
                ix >= 0 &&
                ix < input_width) {

              patch[
                  tap * p.input_channels + c] =
                  input[
                      (iy * input_width + ix)
                      * p.input_channels + c];

            } else {

              patch[
                  tap * p.input_channels + c] =
                  static_cast<int8_t>(
                      input_zero_point);
            }
          }
        }
      }


      // ------------------------------------------------------
      // Execute CFU
      // ------------------------------------------------------

      if (!Mnv2FusedCfu(
              p,
              patch,
              output +
                  (oy * ow + ox)
                  * p.output_channels)) {

        return false;
      }
    }
  }


  return true;
}


// ============================================================
// Hardware CFU driver
// ============================================================

bool Mnv2FusedCfu(
    const Mnv2FusedParams& p,
    const int8_t* input_patch,
    int8_t* output) {

  if (!Valid(p, input_patch, output)) {
    return false;
  }


  // ----------------------------------------------------------
  // RESET
  // ----------------------------------------------------------

  MNV2_FUSED_OP(
      MNV2_FUSED_RESET,
      0);


  // ----------------------------------------------------------
  // CONFIG
  // ----------------------------------------------------------

  MNV2_FUSED_OP(
      MNV2_FUSED_CONFIG,
      MNV2_FUSED_PACK_CONFIG(
          p.input_channels,
          p.expanded_channels,
          p.output_channels));


  // ----------------------------------------------------------
  // SHIFTS
  // ----------------------------------------------------------

  MNV2_FUSED_OP(
      MNV2_FUSED_SET_SHIFTS,
      MNV2_FUSED_PACK_SHIFTS(
          p.expansion_shift,
          p.depthwise_shift,
          p.projection_shift));


  // ----------------------------------------------------------
  // Input patch
  // ----------------------------------------------------------

  for (int n = 0;
       n <
       MNV2_FUSED_KERNEL_TAPS *
       p.input_channels;
       ++n) {

    MNV2_FUSED_OP(
        MNV2_FUSED_PUSH_INPUT,
        static_cast<uint8_t>(
            input_patch[n]));
  }


  // ----------------------------------------------------------
  // Expansion weights
  // ----------------------------------------------------------

  for (int n = 0;
       n <
       p.expanded_channels *
       p.input_channels;
       ++n) {

    MNV2_FUSED_OP(
        MNV2_FUSED_PUSH_EXP_WEIGHT,
        static_cast<uint8_t>(
            p.expansion_weights[n]));
  }


  // ----------------------------------------------------------
  // Expansion bias
  // ----------------------------------------------------------

  for (int n = 0;
       n < p.expanded_channels;
       ++n) {

    MNV2_FUSED_OP(
        MNV2_FUSED_PUSH_EXP_BIAS,
        static_cast<uint32_t>(
            p.expansion_bias[n]));
  }


  // ----------------------------------------------------------
  // Depthwise weights
  // ----------------------------------------------------------

  for (int n = 0;
       n <
       MNV2_FUSED_KERNEL_TAPS *
       p.expanded_channels;
       ++n) {

    MNV2_FUSED_OP(
        MNV2_FUSED_PUSH_DW_WEIGHT,
        static_cast<uint8_t>(
            p.depthwise_weights[n]));
  }


  // ----------------------------------------------------------
  // Depthwise bias
  // ----------------------------------------------------------

  for (int n = 0;
       n < p.expanded_channels;
       ++n) {

    MNV2_FUSED_OP(
        MNV2_FUSED_PUSH_DW_BIAS,
        static_cast<uint32_t>(
            p.depthwise_bias[n]));
  }


  // ----------------------------------------------------------
  // Projection weights
  // ----------------------------------------------------------

  for (int n = 0;
       n <
       p.output_channels *
       p.expanded_channels;
       ++n) {

    MNV2_FUSED_OP(
        MNV2_FUSED_PUSH_PROJ_WEIGHT,
        static_cast<uint8_t>(
            p.projection_weights[n]));
  }


  // ----------------------------------------------------------
  // Projection bias
  // ----------------------------------------------------------

  for (int n = 0;
       n < p.output_channels;
       ++n) {

    MNV2_FUSED_OP(
        MNV2_FUSED_PUSH_PROJ_BIAS,
        static_cast<uint32_t>(
            p.projection_bias[n]));
  }


  // ----------------------------------------------------------
  // RUN
  // ----------------------------------------------------------

  MNV2_FUSED_OP(
      MNV2_FUSED_RUN,
      0);


  // ----------------------------------------------------------
  // STATUS
  // ----------------------------------------------------------

  const uint32_t status =
      MNV2_FUSED_OP(
          MNV2_FUSED_STATUS,
          0);


  // bit 2 = result valid
  // bit 3 = error

  if ((status & (1u << 3)) != 0) {
    return false;
  }


  if ((status & (1u << 2)) == 0) {
    return false;
  }


  // ----------------------------------------------------------
  // POP outputs
  // ----------------------------------------------------------

  for (int o = 0;
       o < p.output_channels;
       ++o) {

    output[o] =
        static_cast<int8_t>(
            MNV2_FUSED_OP(
                MNV2_FUSED_POP_OUTPUT,
                0));
  }


  return true;
}