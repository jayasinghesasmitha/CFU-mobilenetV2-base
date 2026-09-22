#include <stdint.h>
#include <stdio.h>

#include "fused_depthwise.h"
#include "input_image.h"

namespace {

int8_t QuantizeImageByte(unsigned char value) {
  // The actual MobileNetV2 input tensor is UINT8 with zero point 127. The
  // standalone fused CFU consumes signed int8, so center the demonstration
  // patch at that zero point and saturate the one unrepresentable +128 value.
  int32_t centered = static_cast<int32_t>(value) - 127;
  if (centered > 127) centered = 127;
  return static_cast<int8_t>(centered);
}

void LoadCenterRedPatch(int8_t patch[9]) {
  const unsigned int center_y = input_image_height / 2;
  const unsigned int center_x = input_image_width / 2;
  int index = 0;
  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      const unsigned int y = center_y + dy;
      const unsigned int x = center_x + dx;
      const unsigned int offset =
          (y * input_image_width + x) * input_image_channels;
      patch[index++] = QuantizeImageByte(input_image[offset]);  // R channel.
    }
  }
}

}  // namespace

extern "C" void do_proj_menu(void) {
  static const int8_t expansion_weights[1] = {1};
  static const int32_t expansion_bias[1] = {0};
  static const int8_t depthwise_weights[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  static const int32_t depthwise_bias[1] = {0};
  static const int8_t projection_weights[1] = {1};
  static const int32_t projection_bias[1] = {0};
  const Mnv2FusedParams params = {1, 1, 1, 0, 0, 0,
                                  expansion_weights, expansion_bias,
                                  depthwise_weights, depthwise_bias,
                                  projection_weights, projection_bias};
  int8_t input_patch[9];
  int8_t output[1] = {0};
  LoadCenterRedPatch(input_patch);
  const bool image_ok = input_image_len == 224u * 224u * 3u &&
                        input_image_width == 224 &&
                        input_image_height == 224 &&
                        input_image_channels == 3;
  const bool cfu_ok = image_ok && Mnv2FusedCfu(params, input_patch, output);
  printf("mnv2_cfu_depthwise image: %ux%ux%u UINT8 RGB (%u bytes)\n",
         input_image_width, input_image_height, input_image_channels,
         input_image_len);
  printf("center 3x3 red-channel CFU demo: %s output=%d\n",
         cfu_ok ? "ok" : "error", output[0]);
}

extern "C" void mnv2_menu(void) {
  do_proj_menu();
}
