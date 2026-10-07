// fused_dsc.h -- TFLM integration of the fused Expansion->Depthwise->Projection CFU.
//
// MobileNetV2 inverted-residual block =  CONV_2D 1x1 -> [PAD] -> DEPTHWISE_CONV_2D 3x3 -> CONV_2D 1x1.
// TFLM executes these three (or four) operators one after another.  The hooks below make the SAME
// graph run unchanged, but:
//   * the expansion conv, the PAD and the depthwise conv are "captured" (their arguments are stashed,
//     nothing is computed, F1/F2 are never written);
//   * when the projection conv arrives, the whole block is executed on the fused CFU and the result is
//     written straight into the projection output tensor (so the following ADD sees the right data).
// Blocks are identified by their position in the graph (src/fused_plan.h, produced by
// tools/gen_fused_plan.py) and every stashed call is shape-checked.
#pragma once
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace fused {

bool OnConv(const ConvParams& params, const int32_t* output_multiplier, const int32_t* output_shift,
            const RuntimeShape& input_shape, const int8_t* input_data, const RuntimeShape& filter_shape,
            const int8_t* filter_data, const RuntimeShape& bias_shape, const int32_t* bias_data,
            const RuntimeShape& output_shape, int8_t* output_data);

bool OnDepthwise(const DepthwiseParams& params, const int32_t* output_multiplier,
                 const int32_t* output_shift, const RuntimeShape& input_shape, const int8_t* input_data,
                 const RuntimeShape& filter_shape, const int8_t* filter_data, const RuntimeShape& bias_shape,
                 const int32_t* bias_data, const RuntimeShape& output_shape, int8_t* output_data);

bool OnPad();                       // true -> the PAD operator must be skipped

void SetEnabled(bool enabled);      // default: enabled
void ResetStats();
void PrintReport();                 // statistics table (after tflite_classify)

}  // namespace fused
}  // namespace tflite
