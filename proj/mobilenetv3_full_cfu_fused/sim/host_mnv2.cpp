// host_mnv2.cpp -- runs the REAL TFLM interpreter on the PC with the project's patched kernels, and answers
// every fused-CFU custom instruction by clocking the REAL RTL (cfu.v) through Verilator.
//
//   ./host_mnv2 <model.tflite> (--image img.dat | --rand SEED) [--fused 0|1] [--plan plan.txt] [--out out.bin]
//
//   --image  raw 224*224*3 RGB bytes (uint8)      --rand  deterministic pseudo-random input (any model)
//   --plan   fusion plan of a model other than the one compiled into src/fused_plan.h (synthetic test models)
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

static VCfu* top = nullptr;
static uint64_t g_ticks = 0, g_ops = 0, g_by_op[8] = {0};
static inline void tick() { top->clk = 0; top->eval(); top->clk = 1; top->eval(); g_ticks++; }

extern "C" uint32_t dsc_cfu_stub(int f7, uint32_t a, uint32_t b) {
  top->cmd_payload_function_id = (uint32_t)f7 << 3;
  top->cmd_payload_inputs_0 = a; top->cmd_payload_inputs_1 = b;
  top->cmd_valid = 1; top->rsp_ready = 1;
  while (!top->cmd_ready) tick();
  tick();
  top->cmd_valid = 0;
  while (!top->rsp_valid) tick();
  uint32_t r = top->rsp_payload_outputs_0;
  tick();
  g_ops++; if (f7 >= 0 && f7 < 8) g_by_op[f7]++;
  return r;
}

static uint32_t fnv1a(const unsigned char* d, size_t n) {
  uint32_t h = 2166136261u; for (size_t i = 0; i < n; i++) { h ^= d[i]; h *= 16777619u; } return h;
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s model.tflite (--image f.dat | --rand SEED) [--fused 0|1] [--plan p.txt] [--out f]\n", argv[0]); return 1; }
  bool fused = true; const char *outfile = nullptr, *imagefile = nullptr, *planfile = nullptr; long seed = -1;
  for (int i = 2; i < argc; i++) {
    if (!strcmp(argv[i], "--fused") && i + 1 < argc) fused = atoi(argv[++i]) != 0;
    else if (!strcmp(argv[i], "--out") && i + 1 < argc) outfile = argv[++i];
    else if (!strcmp(argv[i], "--image") && i + 1 < argc) imagefile = argv[++i];
    else if (!strcmp(argv[i], "--plan") && i + 1 < argc) planfile = argv[++i];
    else if (!strcmp(argv[i], "--rand") && i + 1 < argc) seed = atol(argv[++i]);
  }
  FILE* f = fopen(argv[1], "rb"); if (!f) { perror("model"); return 1; }
  fseek(f, 0, SEEK_END); long len = ftell(f); fseek(f, 0, SEEK_SET);
  std::vector<uint8_t> model(len); if (fread(model.data(), 1, len, f) != (size_t)len) return 1; fclose(f);

  std::vector<tflite::fused::PlanBlock> plan;
  if (planfile) {
    FILE* pf = fopen(planfile, "r"); if (!pf) { perror("plan"); return 1; }
    tflite::fused::PlanBlock b;
    while (fscanf(pf, "%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d", &b.e, &b.hs1, &b.pad, &b.d, &b.hs2, &b.mean,
                  &b.mulg, &b.p, &b.H, &b.W, &b.N, &b.M, &b.P, &b.K, &b.stride, &b.oh, &b.ow, &b.pad_top, &b.pad_left) == 19) plan.push_back(b);
    fclose(pf);
    tflite::fused::SetPlanForTest(plan.data(), (int)plan.size());
  }

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
  if (imagefile) {
    std::vector<uint8_t> img(in->bytes);
    FILE* g = fopen(imagefile, "rb"); if (!g || fread(img.data(), 1, img.size(), g) != img.size()) { fprintf(stderr, "bad image file\n"); return 1; } fclose(g);
    if (in->type == kTfLiteUInt8) memcpy(in->data.uint8, img.data(), in->bytes);
    else for (size_t i = 0; i < in->bytes; i++) in->data.int8[i] = (int8_t)((int)img[i] - 128);
  } else if (seed >= 0) {
    uint64_t x = 88172645463325252ull ^ (uint64_t)seed * 2654435761ull;
    for (size_t i = 0; i < in->bytes; i++) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; in->data.uint8[i] = (uint8_t)(x >> 24); }
  } else { fprintf(stderr, "need --image or --rand\n"); return 1; }

  tflite::fused::SetEnabled(fused);
  tflite::fused::ResetStats();
  auto t0 = std::chrono::steady_clock::now();
  if (interp.Invoke() != kTfLiteOk) { puts("Invoke failed"); return 1; }
  double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  TfLiteTensor* out = interp.output(0);
  const unsigned char* raw = (const unsigned char*)out->data.raw;
  printf("\nmode=%s   output type=%d bytes=%zu\n", fused ? "FUSED" : "PLAIN", (int)out->type, out->bytes);
  if (out->type == kTfLiteInt8) {
    const int8_t* q = out->data.int8; int n = (int)out->bytes, best = 0, mn = q[0], mx = q[0]; long sum = 0;
    for (int i = 0; i < n; i++) { if (q[i] > q[best]) best = i; if (q[i] < mn) mn = q[i]; if (q[i] > mx) mx = q[i]; sum += q[i]; }
    const float sc = out->params.scale; const int zp = out->params.zero_point;
    printf("top1=%d  q=%d  value=%ld x10^-6   min=%d max=%d mean_q=%ld  distinct-ish span=%d\n", best, q[best],
           (long)((q[best] - zp) * sc * 1e6f), mn, mx, sum / n, mx - mn);
    if (n == 1000) {
      printf("first outputs (x10^-6):"); for (int i = 0; i < 8; i++) printf(" %ld", (long)((q[i] - zp) * sc * 1e6f)); printf("\n");
    }
  }
  printf("output FNV1a = 0x%08x   wall %.1f s   RTL clock ticks=%llu  CFU instructions=%llu\n", fnv1a(raw, out->bytes),
         secs, (unsigned long long)g_ticks, (unsigned long long)g_ops);
  if (outfile) { FILE* w = fopen(outfile, "wb"); fwrite(raw, 1, out->bytes, w); fclose(w); }
  if (fused) {
    printf("CFU instructions by type: CFG=%llu MEMW=%llu START=%llu STATUS=%llu OUTRD=%llu CYCLES=%llu\n",
           (unsigned long long)g_by_op[1], (unsigned long long)g_by_op[2], (unsigned long long)g_by_op[3],
           (unsigned long long)g_by_op[4], (unsigned long long)g_by_op[5], (unsigned long long)g_by_op[7]);
    tflite::fused::PrintReport();
  }
  return 0;
}
