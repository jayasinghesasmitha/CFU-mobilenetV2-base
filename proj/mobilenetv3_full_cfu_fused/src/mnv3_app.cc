// mnv3_app.cc -- MobileNetV3-small INT8 (full: 5x5 depthwise, hard-swish, squeeze-and-excite) on the fused DSC CFU.
// Prints the same sections and labels as the MobileNetV2 projects (mnv2_cfu_package / fused_cfu_with_mobilenetv2),
// so tools/compare_runs.py works on the logs of all of them.
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "cat_image.h"
#include "fused_dsc.h"
#include "models/mnv2/mobilenetv3_small_full_int8.h"
#include "tflite.h"

// Output tensor of this model: INT8 softmax [1,1000], scale = 1/256, zero point = -128
//   probability = (q + 128) / 256      ->  printed as  (q + 128) * 10^6 / 256   ("x 10^-6")
static const int kClasses = 1000;
static inline long scaled(int8_t q) { return ((long)q + 128) * 1000000L / 256L; }

static uint32_t fnv1a(const unsigned char* data, size_t len) {
  uint32_t hash = 2166136261u;
  for (size_t i = 0; i < len; ++i) { hash ^= data[i]; hash *= 16777619u; }
  return hash;
}

static void verify_embedded_image() {
  printf("\n========================================\n");
  printf(" Embedded Image Verification\n");
  printf("========================================\n");
  const unsigned int expected_size = 224 * 224 * 3;
  printf("Header length : %u bytes\n", cat_image_len);
  printf("Expected      : %u bytes\n", expected_size);
  printf("Size check    : %s\n", cat_image_len == expected_size ? "OK" : "FAILED");
  unsigned min = 255, max = 0; unsigned long long sum = 0;
  for (unsigned int i = 0; i < cat_image_len; ++i) {
    if (cat_image[i] < min) min = cat_image[i];
    if (cat_image[i] > max) max = cat_image[i];
    sum += cat_image[i];
  }
  printf("RGB Min       : %u\n", min);
  printf("RGB Max       : %u\n", max);
  printf("RGB Mean      : %lu x 10^-6\n", (unsigned long)(sum * 1000000ULL / cat_image_len));
  printf("RGB FNV1a     : 0x%08lx\n", (unsigned long)fnv1a(cat_image, cat_image_len));
}

static void print_top_outputs() {
  const int8_t* q = tflite_get_output();
  if (q == nullptr) { printf("\nERROR: Could not obtain INT8 output tensor.\n"); return; }

  long min_v = scaled(q[0]), max_v = min_v, sum = 0;
  int best = 0;
  for (int i = 0; i < kClasses; ++i) {
    long v = scaled(q[i]);
    if (v < min_v) min_v = v;
    if (v > max_v) max_v = v;
    if (q[i] > q[best]) best = i;
    sum += v;
  }
  printf("\n========================================\n");
  printf(" MobileNetV3 Classification Result\n");
  printf("========================================\n");
  printf("Output tensor : INT8 [1,1000], probability = (q + 128) / 256\n");
  printf("\nOutput statistics:\n");
  printf("  Min  = %ld x 10^-6\n", min_v);
  printf("  Max  = %ld x 10^-6\n", max_v);
  printf("  Mean = %ld x 10^-6\n", sum / kClasses);
  printf("\nTop-1 class index : %d\n", best);
  printf("Top-1 score       : %ld x 10^-6\n", scaled(q[best]));
  printf("\nFirst 20 outputs:\n");
  for (int i = 0; i < 20; ++i) printf("  output[%d] = %ld x 10^-6\n", i, scaled(q[i]));
}

void mnv3_run(void) {
  printf("\n========================================\n");
  printf(" MobileNetV3-small (full) INT8\n");
  printf(" FUSED-CFU VERSION (3x3/5x5 depthwise, hard-swish, squeeze-excite)\n");
  printf("========================================\n");
  printf("Model             : mobilenetv3_small_full_int8.tflite\n");
  printf("Input             : embedded image\n");
  printf("Resolution        : 224 x 224 x 3\n");
  printf("External input    : UINT8 (converted to INT8 by -128)\n");
  printf("Internal compute  : INT8\n");
  printf("External output   : INT8 softmax\n");
  printf("Classes           : 1000\n");
  printf("Mode              : FUSED DSC CFU (Ex-Dw-Pr, K=3|5, HS, SE)\n");
  printf("Accelerated op    : Ex->Dw->Pr inverted residual blocks (incl. squeeze-and-excite)\n");
  printf("========================================\n");

  printf("\nLoading MobileNetV3 model...\n");
  tflite_load_model(mobilenetv3_small_full_int8, mobilenetv3_small_full_int8_len);
  printf("Model loaded successfully.\n");

  verify_embedded_image();

  printf("\nLoading input image into TFLite...\n");
  tflite_set_input_unsigned(cat_image);
  printf("Input image loaded successfully.\n");

  printf("\nRunning MobileNetV3 inference...\n");
  printf("----------------------------------------\n");
  tflite::fused::ResetStats();
  tflite_classify();

  print_top_outputs();
  tflite::fused::PrintReport();
  printf("\nInference completed.\n");
}

extern "C" void do_proj_menu(void) { mnv3_run(); }
extern "C" void mnv3_menu(void) { mnv3_run(); }
extern "C" void mnv2_menu(void) { mnv3_run(); }
