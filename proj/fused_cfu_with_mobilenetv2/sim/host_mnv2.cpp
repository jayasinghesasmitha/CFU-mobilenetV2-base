// host_mnv2.cpp -- runs the REAL TFLM interpreter on the PC with the project's patched kernels, and answers
// every fused-CFU custom instruction by clocking the REAL RTL (cfu.v) through Verilator.
//
//   ./host_mnv2 <model.tflite> [--fused 0|1] [--verify] [--out file.bin]
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <vector>

#include "VCfu.h"
#include "verilated.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "cat_image.h"
#include "fused_dsc.h"

// ------------------------------------------------------------------ RTL bridge
static VCfu* top = nullptr;
static uint64_t g_ticks = 0, g_ops = 0, g_by_op[8] = {0};
static inline void tick() { top->clk = 0; top->eval(); top->clk = 1; top->eval(); g_ticks++; }

extern "C" uint32_t dsc_cfu_stub(int f7, uint32_t a, uint32_t b) {
  top->cmd_payload_function_id = (uint32_t)f7 << 3;
  top->cmd_payload_inputs_0 = a; top->cmd_payload_inputs_1 = b;
  top->cmd_valid = 1; top->rsp_ready = 1;
  while (!top->cmd_ready) tick();
  tick();                                   // command accepted on this edge
  top->cmd_valid = 0;
  while (!top->rsp_valid) tick();
  uint32_t r = top->rsp_payload_outputs_0;
  tick();                                   // response consumed
  g_ops++; if (f7 >= 0 && f7 < 8) g_by_op[f7]++;
  return r;
}

static uint32_t fnv1a(const unsigned char* d, size_t n) {
  uint32_t h = 2166136261u; for (size_t i = 0; i < n; i++) { h ^= d[i]; h *= 16777619u; } return h;
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s model.tflite [--fused 0|1] [--out f]\n", argv[0]); return 1; }
  bool fused = true; const char* outfile = nullptr;
  for (int i = 2; i < argc; i++) {
    if (!strcmp(argv[i], "--fused") && i + 1 < argc) fused = atoi(argv[++i]) != 0;
    else if (!strcmp(argv[i], "--out") && i + 1 < argc) outfile = argv[++i];
  }
  FILE* f = fopen(argv[1], "rb"); if (!f) { perror("model"); return 1; }
  fseek(f, 0, SEEK_END); long len = ftell(f); fseek(f, 0, SEEK_SET);
  std::vector<uint8_t> model(len); if (fread(model.data(), 1, len, f) != (size_t)len) return 1; fclose(f);

  Verilated::commandArgs(argc, argv);
  top = new VCfu; top->reset = 1; top->cmd_valid = 0; top->rsp_ready = 0;
  for (int i = 0; i < 8; i++) tick();
  top->reset = 0; for (int i = 0; i < 4; i++) tick();
  printf("CFU PING -> 0x%08x\n", dsc_cfu_stub(0, 0, 0));

  const size_t kArena = 48u << 20;
  uint8_t* arena = (uint8_t*)aligned_alloc(16, kArena);
  tflite::AllOpsResolver resolver;
  tflite::MicroInterpreter interp(tflite::GetModel(model.data()), resolver, arena, kArena);
  if (interp.AllocateTensors() != kTfLiteOk) { puts("AllocateTensors failed"); return 1; }
  TfLiteTensor* in = interp.input(0);
  printf("input type=%d bytes=%zu   arena used=%zu bytes\n", (int)in->type, in->bytes, interp.arena_used_bytes());
  if (in->type == kTfLiteUInt8) memcpy(in->data.uint8, cat_image, in->bytes);
  else for (size_t i = 0; i < in->bytes; i++) in->data.int8[i] = (int8_t)((int)cat_image[i] - 128);

  tflite::fused::SetEnabled(fused);
  tflite::fused::ResetStats();
  auto t0 = std::chrono::steady_clock::now();
  if (interp.Invoke() != kTfLiteOk) { puts("Invoke failed"); return 1; }
  double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  TfLiteTensor* out = interp.output(0);
  const float* o = out->data.f; int n = (int)(out->bytes / sizeof(float));
  int best = 0; float mn = o[0], mx = o[0]; double sum = 0;
  for (int i = 0; i < n; i++) { if (o[i] > o[best]) best = i; if (o[i] < mn) mn = o[i]; if (o[i] > mx) mx = o[i]; sum += o[i]; }
  printf("\nmode=%s   outputs=%d   top1=%d   score=%ld x10^-6   min=%ld max=%ld mean=%ld (x10^-6)\n",
         fused ? "FUSED" : "PLAIN", n, best, (long)(o[best] * 1e6f), (long)(mn * 1e6f), (long)(mx * 1e6f), (long)(sum / n * 1e6));
  printf("output FNV1a = 0x%08x   wall %.1f s   RTL clock ticks=%llu  CFU instructions=%llu\n",
         fnv1a((const unsigned char*)o, n * sizeof(float)), secs, (unsigned long long)g_ticks, (unsigned long long)g_ops);
  if (outfile) { FILE* w = fopen(outfile, "wb"); fwrite(o, sizeof(float), n, w); fclose(w); }
  if (fused) {
    printf("CFU instructions by type: CFG=%llu MEMW=%llu START=%llu STATUS=%llu OUTRD=%llu CYCLES=%llu\n",
           (unsigned long long)g_by_op[1], (unsigned long long)g_by_op[2], (unsigned long long)g_by_op[3],
           (unsigned long long)g_by_op[4], (unsigned long long)g_by_op[5], (unsigned long long)g_by_op[7]);
    tflite::fused::PrintReport();
  }
  return 0;
}
