// mnv2_webcam_app.cc -- MobileNetV2 a0.35 INT8 on the fused DSC CFU, fed with images from a PC webcam program.
//
// The bitstream and this firmware are loaded ONCE.  The firmware then waits on the UART for 224x224 RGB images
// (see image_server.h), classifies each one on the fused CFU and prints the result.  No rebuild per picture.
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "fused_dsc.h"
#include "image_server.h"
#include "models/mnv2/mobilenetv2_a035_224_int8.h"
#include "tflite.h"

static const float* infer_one(const uint8_t* rgb) {
  tflite_set_input_unsigned(rgb);              // UINT8 RGB -> TFLite input tensor (model quantises internally)
  tflite::fused::ResetStats();
  tflite_classify();                           // prints the per-op profile and "( N ) cycles total"
  int blocks = 0; uint32_t hw = 0, cpu = 0;
  tflite::fused::GetTotals(&blocks, &hw, &cpu);
  printf("@@FUSED blocks=%d hw_cycles=%lu cpu_cycles=%lu\n", blocks, (unsigned long)hw, (unsigned long)cpu);
  return tflite_get_output_float();
}

static void mnv2_webcam_run(void) {
  printf("\n========================================\n");
  printf(" MobileNetV2 a0.35 INT8  -  FUSED-CFU WEBCAM SERVER\n");
  printf("========================================\n");
  printf("Model             : mobilenetv2_a035_224_int8.tflite\n");
  printf("Input             : 224 x 224 x 3 UINT8 RGB, received over UART\n");
  printf("Mode              : FUSED DSC CFU (Ex-Dw-Pr)\n");
  printf("Accelerated op    : Ex->Dw->Pr inverted residual blocks\n");
  printf("========================================\n");
  printf("Loading MobileNetV2 model...\n");
  tflite_load_model(mobilenetv2_a035_224_int8, mobilenetv2_a035_224_int8_len);
  printf("Model loaded successfully.\n");
  imgsrv::Serve(infer_one, "fused_dsc_cfu");
}

extern "C" void do_proj_menu(void) { mnv2_webcam_run(); }
extern "C" void mnv2_menu(void) { mnv2_webcam_run(); }
