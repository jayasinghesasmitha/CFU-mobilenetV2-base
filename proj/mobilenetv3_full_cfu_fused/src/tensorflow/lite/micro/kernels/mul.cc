/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/micro/kernels/mul.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/mul.h"
#include "tensorflow/lite/kernels/internal/reference/mul.h"
#include "tensorflow/lite/kernels/internal/reference/process_broadcast_shapes.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "fused_dsc.h"   // FUSED-CFU PATCH

namespace tflite {

TfLiteStatus MulEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  auto* params = reinterpret_cast<TfLiteMulParams*>(node->builtin_data);

  TFLITE_DCHECK(node->user_data != nullptr);
  const OpDataMul* data = static_cast<const OpDataMul*>(node->user_data);

  const TfLiteEvalTensor* input1 =
      tflite::micro::GetEvalInput(context, node, kMulInput1Tensor);
  const TfLiteEvalTensor* input2 =
      tflite::micro::GetEvalInput(context, node, kMulInput2Tensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kMulOutputTensor);

  // FUSED-CFU PATCH: the gating MUL of a squeeze-and-excite block is folded into the fused block
  if (input1->type == kTfLiteInt8 &&
      tflite::fused::OnMul(data->input1_zero_point, data->input2_zero_point, data->output_zero_point,
                           data->output_multiplier, data->output_shift, data->output_activation_min,
                           data->output_activation_max, input1->dims->data, input1->dims->size,
                           tflite::micro::GetTensorData<int8_t>(input1), input2->dims->data, input2->dims->size,
                           tflite::micro::GetTensorData<int8_t>(input2)))
    return kTfLiteOk;
  switch (input1->type) {
    case kTfLiteInt8:
    case kTfLiteInt16:
    case kTfLiteInt32:
      EvalMulQuantizedReference(context, node, data, input1, input2, output);
      break;
    case kTfLiteFloat32:
      EvalMulFloatReference(context, node, params, data, input1, input2,
                            output);
      break;
    default:
      MicroPrintf("Type %s (%d) not supported.",
                  TfLiteTypeGetName(input1->type), input1->type);
      return kTfLiteError;
  }

  return kTfLiteOk;
}

TfLiteRegistration Register_MUL() {
  return tflite::micro::RegisterOp(MulInit, MulPrepare, MulEval);
}

}  // namespace tflite
