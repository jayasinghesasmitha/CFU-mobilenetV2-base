#ifndef PROJ_MNV2_CFU_DEPTHWISE_FUSED_DEPTHWISE_H_
#define PROJ_MNV2_CFU_DEPTHWISE_FUSED_DEPTHWISE_H_

#include <stdint.h>

struct Mnv2FusedParams {
  int input_channels;
  int expanded_channels;
  int output_channels;
  int expansion_shift;
  int depthwise_shift;
  int projection_shift;

  // Layouts: [expanded][input], [tap][expanded], [output][expanded].
  const int8_t* expansion_weights;
  const int32_t* expansion_bias;
  const int8_t* depthwise_weights;
  const int32_t* depthwise_bias;
  const int8_t* projection_weights;
  const int32_t* projection_bias;
};

// Computes one output pixel from a flattened [9][input_channels] patch.
// Each stage applies signed arithmetic right shift followed by int8 saturation.
bool Mnv2FusedReference(const Mnv2FusedParams& params,
                        const int8_t* input_patch,
                        int8_t* output);

// Executes the same one-pixel operation through the explicit CPU-fed CFU
// command stream. It performs no DMA and does not inspect TFLite tensors.
bool Mnv2FusedCfu(const Mnv2FusedParams& params,
                  const int8_t* input_patch,
                  int8_t* output);

// Runs a complete single-batch NHWC image without materializing Expansion or
// Depthwise intermediate tensors. Each output pixel creates a bounded local
// 3x3 patch, streams it through the fused CFU, and writes only the final
// projection result. `padding` is 0 (VALID) or 1 (SAME for a 3x3 kernel), and
// `stride` is 1 or 2. Out-of-bounds patch values use input_zero_point.
bool Mnv2FusedRunImage(const Mnv2FusedParams& params,
                       const int8_t* input,
                       int input_height,
                       int input_width,
                       int stride,
                       int padding,
                       int input_zero_point,
                       int8_t* output,
                       int output_capacity,
                       int* output_height,
                       int* output_width);

#endif  // PROJ_MNV2_CFU_DEPTHWISE_FUSED_DEPTHWISE_H_
