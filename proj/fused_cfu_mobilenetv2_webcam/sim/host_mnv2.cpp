// host_mnv2.cpp -- runs the REAL TFLM interpreter on the PC with the project's patched kernels, and answers
// every fused-CFU custom instruction by clocking the REAL RTL (cfu.v) through Verilator.
//
//   ./host_mnv2 <model.tflite> --image img.dat [--fused 0|1] [--out file.bin]     one image, print result
//   ./host_mnv2 <model.tflite> --serve [--fused 0|1]                                speak the UART image protocol on stdin/stdout
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

#include "fused_dsc.h"
#include "image_server.h"

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

static tflite::MicroInterpreter* g_interp = nullptr;
static bool g_fused = true;
static const float* serve_infer(const uint8_t* rgb) {
  TfLiteTensor* in = g_interp->input(0);
  if (in->type == kTfLiteUInt8) memcpy(in->data.uint8, rgb, in->bytes);
  else for (size_t i = 0; i < in->bytes; i++) in->data.int8[i] = (int8_t)((int)rgb[i] - 128);
  tflite::fused::ResetStats();
  if (g_interp->Invoke() != kTfLiteOk) return nullptr;
  int blocks = 0; uint32_t hw = 0, cpu = 0;
  tflite::fused::GetTotals(&blocks, &hw, &cpu);
  if (g_fused) printf("@@FUSED blocks=%d hw_cycles=%lu cpu_cycles=%lu\n", blocks, (unsigned long)hw, (unsigned long)cpu);
  return g_interp->output(0)->data.f;
}

static uint32_t fnv1a(const unsigned char* d, size_t n) {
  uint32_t h = 2166136261u; for (size_t i = 0; i < n; i++) { h ^= d[i]; h *= 16777619u; } return h;
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s model.tflite [--fused 0|1] [--out f]\n", argv[0]); return 1; }
  bool fused = true, serve = false; const char* outfile = nullptr; const char* imagefile = nullptr;
  for (int i = 2; i < argc; i++) {
    if (!strcmp(argv[i], "--fused") && i + 1 < argc) fused = atoi(argv[++i]) != 0;
    else if (!strcmp(argv[i], "--out") && i + 1 < argc) outfile = argv[++i];
    else if (!strcmp(argv[i], "--image") && i + 1 < argc) imagefile = argv[++i];
    else if (!strcmp(argv[i], "--serve")) serve = true;
  }
  FILE* f = fopen(argv[1], "rb"); if (!f) { perror("model"); return 1; }
  fseek(f, 0, SEEK_END); long len = ftell(f); fseek(f, 0, SEEK_SET);
  std::vector<uint8_t> model(len); if (fread(model.data(), 1, len, f) != (size_t)len) return 1; fclose(f);

  Verilated::commandArgs(argc, argv);
  top = new VCfu; top->reset = 1; top->cmd_valid = 0; top->rsp_ready = 0;
  for (int i = 0; i < 8; i++) tick();
  top->reset = 0; for (int i = 0; i < 4; i++) tick();
  uint32_t ping = dsc_cfu_stub(0, 0, 0);
  if (!serve) printf("CFU PING -> 0x%08x\n", ping);

  const size_t kArena = 48u << 20;
  uint8_t* arena = (uint8_t*)aligned_alloc(16, kArena);
  tflite::AllOpsResolver resolver;
  tflite::MicroInterpreter interp(tflite::GetModel(model.data()), resolver, arena, kArena);
  if (interp.AllocateTensors() != kTfLiteOk) { puts("AllocateTensors failed"); return 1; }
  g_interp = &interp; g_fused = fused;
  tflite::fused::SetEnabled(fused);
  if (serve) { imgsrv::Serve(serve_infer, fused ? "host_sim_fused_rtl" : "host_sim_plain"); return 0; }
  if (!imagefile) { fprintf(stderr, "need --image file.dat or --serve\n"); return 1; }
  TfLiteTensor* in = interp.input(0);
  printf("input type=%d bytes=%zu   arena used=%zu bytes\n", (int)in->type, in->bytes, interp.arena_used_bytes());
  std::vector<uint8_t> img(in->bytes);
  { FILE* g = fopen(imagefile, "rb"); if (!g || fread(img.data(), 1, img.size(), g) != img.size()) { fprintf(stderr, "bad image file\n"); return 1; } fclose(g); }
  if (in->type == kTfLiteUInt8) memcpy(in->data.uint8, img.data(), in->bytes);
  else for (size_t i = 0; i < in->bytes; i++) in->data.int8[i] = (int8_t)((int)img[i] - 128);

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
