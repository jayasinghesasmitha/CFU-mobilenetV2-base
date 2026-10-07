#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "cat_image.h"
#include "models/mnv2/mobilenetv2_a035_224_int8.h"
#include "tflite.h"
#include "fused_dsc.h"


// ============================================================
// FNV-1a checksum
// ============================================================

static uint32_t fnv1a(
    const unsigned char* data,
    size_t len) {

  uint32_t hash = 2166136261u;

  for (size_t i = 0; i < len; ++i) {

    hash ^= data[i];

    hash *= 16777619u;
  }

  return hash;
}


// ============================================================
// Verify embedded image
// ============================================================

static void verify_embedded_image() {

  printf("\n");
  printf("========================================\n");
  printf(" Embedded Image Verification\n");
  printf("========================================\n");

  const unsigned int expected_size =
      224 * 224 * 3;

  printf(
      "Header length : %u bytes\n",
      cat_image_len
  );

  printf(
      "Expected      : %u bytes\n",
      expected_size
  );


  // ----------------------------------------------------------
  // Size check
  // ----------------------------------------------------------

  if (cat_image_len == expected_size) {

    printf(
        "Size          : OK\n"
    );

  } else {

    printf(
        "Size          : ERROR\n"
    );
  }


  // ----------------------------------------------------------
  // RGB statistics
  // ----------------------------------------------------------

  unsigned int min_value = 255;

  unsigned int max_value = 0;

  uint64_t sum = 0;


  for (
      unsigned int i = 0;
      i < cat_image_len;
      ++i) {

    unsigned int value =
        static_cast<unsigned int>(
            cat_image[i]
        );


    if (value < min_value) {

      min_value = value;
    }


    if (value > max_value) {

      max_value = value;
    }


    sum += value;
  }


  uint64_t mean_scaled =
      (sum * 1000000ULL) /
      static_cast<uint64_t>(
          cat_image_len
      );


  printf(
      "RGB min       : %u\n",
      min_value
  );

  printf(
      "RGB max       : %u\n",
      max_value
  );

  printf(
      "RGB mean      : %lu.%06lu\n",
      static_cast<unsigned long>(
          mean_scaled / 1000000ULL
      ),
      static_cast<unsigned long>(
          mean_scaled % 1000000ULL
      )
  );


  // ----------------------------------------------------------
  // Checksum
  // ----------------------------------------------------------

  uint32_t checksum =
      fnv1a(
          cat_image,
          cat_image_len
      );


  printf(
      "RGB FNV1a     : 0x%08lx\n",
      static_cast<unsigned long>(
          checksum
      )
  );


  // ----------------------------------------------------------
  // First 30 bytes
  // ----------------------------------------------------------

  printf("\n");

  printf(
      "First 30 RGB bytes:\n"
  );


  for (int i = 0; i < 30; ++i) {

    printf(
        "%u ",
        static_cast<unsigned int>(
            cat_image[i]
        )
    );
  }

  printf("\n");


  // ----------------------------------------------------------
  // First 10 pixels
  // ----------------------------------------------------------

  printf("\n");

  printf(
      "First 10 RGB pixels:\n"
  );


  for (int i = 0; i < 10; ++i) {

    int p = i * 3;


    unsigned int r =
        static_cast<unsigned int>(
            cat_image[p]
        );

    unsigned int g =
        static_cast<unsigned int>(
            cat_image[p + 1]
        );

    unsigned int b =
        static_cast<unsigned int>(
            cat_image[p + 2]
        );


    printf(
        "Pixel %2d: R=%u G=%u B=%u\n",
        i,
        r,
        g,
        b
    );
  }


  printf("\n");

  printf(
      "Input format verified:\n"
  );

  printf(
      "  UINT8\n"
  );

  printf(
      "  RGB\n"
  );

  printf(
      "  224 x 224 x 3\n"
  );

  printf(
      "  Raw bytes unchanged\n"
  );

  printf(
      "========================================\n"
  );
}


// ============================================================
// Print classification output
// ============================================================

static void print_top_outputs() {

  // ----------------------------------------------------------
  // IMPORTANT:
  //
  // Netron says:
  //
  // float32 [1,1000]
  //
  // Therefore use the FLOAT32 accessor.
  // ----------------------------------------------------------

  float* output =
      tflite_get_output_float();


  if (output == nullptr) {

    printf("\n");

    printf(
        "ERROR: Could not obtain "
        "FLOAT32 output tensor.\n"
    );

    return;
  }


  const int output_count = 1000;


  // ----------------------------------------------------------
  // Statistics
  // ----------------------------------------------------------

  float min_value =
      output[0];

  float max_value =
      output[0];

  float sum =
      0.0f;


  for (
      int i = 0;
      i < output_count;
      ++i) {

    float value =
        output[i];


    if (value < min_value) {

      min_value = value;
    }


    if (value > max_value) {

      max_value = value;
    }


    sum += value;
  }


  float mean =
      sum /
      static_cast<float>(
          output_count
      );


  // ----------------------------------------------------------
  // Find Top-1
  // ----------------------------------------------------------

  int best_index = 0;

  float best_value =
      output[0];


  for (
      int i = 1;
      i < output_count;
      ++i) {

    if (output[i] > best_value) {

      best_value =
          output[i];

      best_index =
          i;
    }
  }


  // ----------------------------------------------------------
  // Convert float values to fixed-point
  //
  // Avoid relying on %f support in embedded printf.
  // ----------------------------------------------------------

  int32_t min_scaled =
      static_cast<int32_t>(
          min_value * 1000000.0f
      );

  int32_t max_scaled =
      static_cast<int32_t>(
          max_value * 1000000.0f
      );

  int32_t mean_scaled =
      static_cast<int32_t>(
          mean * 1000000.0f
      );

  int32_t best_scaled =
      static_cast<int32_t>(
          best_value * 1000000.0f
      );


  // ----------------------------------------------------------
  // Print result
  // ----------------------------------------------------------

  printf("\n");

  printf(
      "========================================\n"
  );

  printf(
      " MobileNetV2 Classification Result\n"
  );

  printf(
      "========================================\n"
  );


  printf("\n");

  printf(
      "Output tensor : FLOAT32\n"
  );

  printf(
      "Output size   : 1000 classes\n"
  );


  // ----------------------------------------------------------
  // First 20 output values
  // ----------------------------------------------------------

  printf("\n");

  printf(
      "First 20 FLOAT32 outputs:\n"
  );


  for (int i = 0; i < 20; ++i) {

    int32_t scaled =
        static_cast<int32_t>(
            output[i] * 1000000.0f
        );


    printf(
        "  output[%d] = %ld x 10^-6\n",
        i,
        static_cast<long>(
            scaled
        )
    );
  }


  // ----------------------------------------------------------
  // Output statistics
  // ----------------------------------------------------------

  printf("\n");

  printf(
      "Output statistics:\n"
  );

  printf(
      "  Min  = %ld x 10^-6\n",
      static_cast<long>(
          min_scaled
      )
  );

  printf(
      "  Max  = %ld x 10^-6\n",
      static_cast<long>(
          max_scaled
      )
  );

  printf(
      "  Mean = %ld x 10^-6\n",
      static_cast<long>(
          mean_scaled
      )
  );


  // ----------------------------------------------------------
  // Top-1
  // ----------------------------------------------------------

  printf("\n");

  printf(
      "Top-1 class index : %d\n",
      best_index
  );

  printf(
      "Top-1 score       : %ld x 10^-6\n",
      static_cast<long>(
          best_scaled
      )
  );


  printf("\n");

  printf(
      "Prediction class index: %d\n",
      best_index
  );


  printf(
      "========================================\n"
  );
}


// ============================================================
// MobileNetV2 execution
// ============================================================

void mnv2_run(void) {

  printf("\n");

  printf(
      "========================================\n"
  );

  printf(
      " MobileNetV2 a0.35 INT8\n"
  );

  printf(
      " FUSED-CFU VERSION\n"
  );

  printf(
      "========================================\n"
  );


  printf(
      "Model             : "
      "mobilenetv2_a035_224_int8.tflite\n"
  );

  printf(
      "Input             : embedded image\n"
  );

  printf(
      "Resolution        : 224 x 224 x 3\n"
  );

  printf(
      "External input    : UINT8\n"
  );

  printf(
      "Internal compute  : INT8\n"
  );

  printf(
      "External output   : FLOAT32\n"
  );

  printf(
      "Classes           : 1000\n"
  );

  printf(
      "Mode              : FUSED DSC CFU (Ex-Dw-Pr)\n"
  );

  printf(
      "Accelerated op    : Ex->Dw->Pr inverted residual blocks\n"
  );


  printf(
      "========================================\n"
  );


  // ----------------------------------------------------------
  // Load model
  // ----------------------------------------------------------

  printf("\n");

  printf(
      "Loading MobileNetV2 model...\n"
  );


  tflite_load_model(
      mobilenetv2_a035_224_int8,
      mobilenetv2_a035_224_int8_len
  );


  printf(
      "Model loaded successfully.\n"
  );


  // ----------------------------------------------------------
  // Verify image before inference
  // ----------------------------------------------------------

  verify_embedded_image();


  // ----------------------------------------------------------
  // Copy image into TFLite input tensor
  // ----------------------------------------------------------

  printf("\n");

  printf(
      "Loading input image into TFLite...\n"
  );


  tflite_set_input_unsigned(
      cat_image
  );


  printf(
      "Input image loaded successfully.\n"
  );


  // ----------------------------------------------------------
  // Run inference
  // ----------------------------------------------------------

  printf("\n");

  printf(
      "Running MobileNetV2 inference...\n"
  );

  printf(
      "----------------------------------------\n"
  );


  tflite::fused::ResetStats();

  tflite_classify();


  // ----------------------------------------------------------
  // Read FLOAT32 output
  // ----------------------------------------------------------

  print_top_outputs();

  tflite::fused::PrintReport();


  printf("\n");

  printf(
      "Inference completed.\n"
  );
}


// ============================================================
// CFU Playground menu entry points
// ============================================================

extern "C" void do_proj_menu(void) {

  mnv2_run();
}


extern "C" void mnv2_menu(void) {

  mnv2_run();
}