// fused_dsc.h -- TFLM integration of the fused Expansion -> Depthwise(3x3|5x5) -> Projection CFU.
//
// MobileNet inverted-residual block =  CONV_2D 1x1 -> [HARD_SWISH] -> [PAD] -> DEPTHWISE_CONV_2D 3x3|5x5
//                                      -> [HARD_SWISH] -> [squeeze-and-excite] -> CONV_2D 1x1
//   squeeze-and-excite = MEAN -> CONV_2D -> CONV_2D -> MUL(const) -> MUL(feature map x gate)
// TFLM executes these operators one after another.  The hooks below make the SAME graph run unchanged, but:
//   * the expansion conv, PAD, depthwise conv and HARD_SWISH operators are "captured" (their arguments are
//     stashed, nothing is computed, F1/F2 are never written); hard-swish becomes a 256-entry activation table
//     built with TFLM's own reference function (so it is bit-exact);
//   * when the projection conv arrives, the whole block is executed on the fused CFU and the result is written
//     straight into the projection output tensor (so a following residual ADD sees the right data).
// Blocks are identified by their position in the graph (src/fused_plan.h, produced by tools/gen_fused_plan.py)
// and every captured call is shape-checked.
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

bool OnPad();                                  // true -> the PAD operator must be skipped
bool OnHardSwish(const HardSwishParams& p);    // true -> the (int8) HARD_SWISH operator must be skipped
// squeeze: true -> the MEAN output (in_dims = [1,H,W,C] input dims, out = C values) was produced from the CFU
bool OnMean(int in_zp, float in_scale, int out_zp, float out_scale, const int* in_dims, int8_t* out);
// gate multiply: true -> the gating MUL must be skipped (folded into the projection pass)
bool OnMul(int in1_zp, int in2_zp, int out_zp, int32_t mult, int shift, int32_t amin, int32_t amax,
           const int* dims1, int nd1, const int8_t* d1, const int* dims2, int nd2, const int8_t* d2);

void SetEnabled(bool enabled);                 // default: enabled
void ResetStats();
void PrintReport();                            // statistics table (after tflite_classify)
void GetTotals(int* blocks_run, uint32_t* hw_cycles, uint32_t* cpu_cycles);

// test hook: replace the plan (synthetic tests); nullptr restores the generated plan
struct PlanBlock { int e, hs1, pad, d, hs2, mean, mulg, p, H, W, N, M, P, K, stride, oh, ow, pad_top, pad_left; };
void SetPlanForTest(const PlanBlock* blocks, int n);

}  // namespace fused
}  // namespace tflite
