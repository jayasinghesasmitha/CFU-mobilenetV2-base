// fused_dsc.cc -- see fused_dsc.h
#include "fused_dsc.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsc_driver.h"
#include "fused_plan.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/depthwise_conv.h"

#ifdef FUSED_HOST
static inline uint32_t fused_cycles() { return 0; }
#else
#include "perf.h"
static inline uint32_t fused_cycles() { return (uint32_t)perf_get_mcycle(); }
#endif

namespace tflite {
namespace fused {
namespace {

enum State { kIdle, kGotE, kGotPad, kGotD };

struct ConvStash {
  const int8_t* in; const int8_t* filt; const int32_t* bias; const int32_t* mult; const int32_t* shift;
  int8_t* out;
  int in_h, in_w, in_c, out_h, out_w, out_c;
  int input_offset, output_offset, act_min, act_max;
};
struct DwStash {
  const int8_t* filt; const int32_t* bias; const int32_t* mult; const int32_t* shift; int8_t* out;
  int stride_h, stride_w, pad_h, pad_w, out_h, out_w, out_c;
  int input_offset, output_offset, act_min, act_max;
};

struct BlockStat { uint32_t cpu_cycles, hw_cycles, jobs, strips; int ran; int mismatches; };

bool g_enabled = true;
int g_idx = 0;
State g_state = kIdle;
const fused_block_t* g_blk = nullptr;
int g_blk_no = -1;
ConvStash g_e, g_p;
DwStash g_d;
BlockStat g_stat[sizeof(kFusedBlocks) / sizeof(kFusedBlocks[0])];
uint32_t g_total_cpu = 0;
int g_verify_blocks = 0, g_verify_bad = 0;

// The projection output tensor may share arena memory with the (already dead) expansion input or with F1,
// so the fused result is first written to a private buffer when it overlaps the input.
static int8_t g_scratch_out[32 * 1024];
#ifdef FUSED_VERIFY
// Private buffers for the layer-by-layer cross-check (never alias TFLM tensors).
static int8_t g_vf1[112 * 112 * 48];
static int8_t g_vf2[56 * 56 * 48];
static int8_t g_vref[32 * 1024];
#endif

[[noreturn]] void Fatal(const char* why, int idx) {
  printf("\nFUSED CFU PLAN MISMATCH at tracked op %d: %s\n", idx, why);
  printf("Regenerate src/fused_plan.h with tools/gen_fused_plan.py for this model.\n");
  abort();
}

const fused_block_t* FindByE(int idx, int* no) {
  for (int i = 0; i < kNumFusedBlocks; i++)
    if (kFusedBlocks[i].e == idx) { *no = i; return &kFusedBlocks[i]; }
  return nullptr;
}

void StashConv(ConvStash* s, const ConvParams& p, const int32_t* mult, const int32_t* shift,
               const RuntimeShape& ish, const int8_t* in, const int8_t* filt, const int32_t* bias,
               const RuntimeShape& osh, int8_t* out) {
  s->in = in; s->filt = filt; s->bias = bias; s->mult = mult; s->shift = shift; s->out = out;
  s->in_h = ish.Dims(1); s->in_w = ish.Dims(2); s->in_c = ish.Dims(3);
  s->out_h = osh.Dims(1); s->out_w = osh.Dims(2); s->out_c = osh.Dims(3);
  s->input_offset = p.input_offset; s->output_offset = p.output_offset;
  s->act_min = p.quantized_activation_min; s->act_max = p.quantized_activation_max;
}

void RunBlock() {
  const fused_block_t& b = *g_blk;
  dsc_layer_t L;
  memset(&L, 0, sizeof(L));
  L.H = g_e.in_h; L.W = g_e.in_w; L.N = g_e.in_c; L.M = g_e.out_c; L.P = g_p.out_c;
  L.stride = g_d.stride_h; L.out_h = g_p.out_h; L.out_w = g_p.out_w;
  L.row_off = -g_d.pad_h; L.col_off = -g_d.pad_w;
  L.ex_in_off = g_e.input_offset; L.ex_out_zp = g_e.output_offset; L.ex_min = g_e.act_min; L.ex_max = g_e.act_max;
  L.dw_in_off = g_d.input_offset; L.dw_out_zp = g_d.output_offset; L.dw_min = g_d.act_min; L.dw_max = g_d.act_max;
  L.pr_in_off = g_p.input_offset; L.pr_out_zp = g_p.output_offset; L.pr_min = g_p.act_min; L.pr_max = g_p.act_max;
  L.ex_w = g_e.filt; L.ex_b = g_e.bias; L.ex_m = g_e.mult; L.ex_s = g_e.shift;
  L.dw_w = g_d.filt; L.dw_b = g_d.bias; L.dw_m = g_d.mult; L.dw_s = g_d.shift;
  L.pr_w = g_p.filt; L.pr_b = g_p.bias; L.pr_m = g_p.mult; L.pr_s = g_p.shift;

  if (L.H != b.H || L.W != b.W || L.N != b.N || L.M != b.M || L.P != b.P || L.stride != b.stride ||
      L.out_h != b.oh || L.out_w != b.ow || g_d.out_h != b.oh || g_d.out_w != b.ow)
    Fatal("block dimensions differ from the plan", g_idx - 1);
  if (!dsc_block_supported(&L)) Fatal("block does not fit the CFU limits", g_idx - 1);

  const int n_out = L.out_h * L.out_w * L.P;
  const int n_in = L.H * L.W * L.N;
  const int8_t* in_lo = g_e.in; const int8_t* in_hi = g_e.in + n_in;
  const int8_t* out_lo = g_p.out; const int8_t* out_hi = g_p.out + n_out;
  const bool overlap = (in_lo < out_hi) && (out_lo < in_hi);
  int8_t* dst = g_p.out;
  if (overlap) {
    if (n_out > (int)sizeof(g_scratch_out)) Fatal("output overlaps input and is too large for the scratch buffer", g_idx - 1);
    dst = g_scratch_out;
  }

  const uint32_t hw0 = dsc_stats.hw_cycles, jobs0 = dsc_stats.jobs, strips0 = dsc_stats.strips;
  const uint32_t t0 = fused_cycles();
  const int rc = dsc_run_block(&L, g_e.in, dst);
  if (rc != 0) Fatal("dsc_run_block failed", g_idx - 1);

  BlockStat& s = g_stat[g_blk_no];
  s.hw_cycles = dsc_stats.hw_cycles - hw0; s.jobs = dsc_stats.jobs - jobs0;
  s.strips = dsc_stats.strips - strips0; s.ran = 1;

#ifdef FUSED_VERIFY
  // Cross-check against the layer-by-layer C reference (before the input can be overwritten below).
  if (n_out <= (int)sizeof(g_vref) && L.H * L.W * L.M <= (int)sizeof(g_vf1) && L.out_h * L.out_w * L.M <= (int)sizeof(g_vf2)) {
    dsc_ref_run(&L, g_e.in, g_vf1, g_vf2, g_vref);
    int bad = 0;
    for (int i = 0; i < n_out; i++) bad += (g_vref[i] != dst[i]);
    s.mismatches = bad; g_verify_blocks++; g_verify_bad += bad;
  }
#endif
  if (overlap) memcpy(g_p.out, dst, n_out);
  const uint32_t t1 = fused_cycles();
  s.cpu_cycles = t1 - t0;
  g_total_cpu += s.cpu_cycles;
}

}  // namespace

void SetEnabled(bool e) { g_enabled = e; }

void ResetStats() {
  memset(g_stat, 0, sizeof(g_stat));
  g_total_cpu = 0; g_verify_blocks = 0; g_verify_bad = 0;
  dsc_reset_stats();
}

bool OnConv(const ConvParams& params, const int32_t* mult, const int32_t* shift, const RuntimeShape& ish,
            const int8_t* in, const RuntimeShape& fsh, const int8_t* filt, const RuntimeShape& bsh,
            const int32_t* bias, const RuntimeShape& osh, int8_t* out) {
  (void)bsh;
  if (fsh.Dims(1) == 3 && fsh.Dims(2) == 3 && ish.Dims(3) == 3) {   // network stem: new inference starts
    g_idx = 0; g_state = kIdle; g_blk = nullptr;
  }
  const int idx = g_idx++;
  if (!g_enabled) return false;

  switch (g_state) {
    case kIdle: {
      int no;
      const fused_block_t* b = FindByE(idx, &no);
      if (!b) return false;                                    // ordinary convolution
      if (fsh.Dims(1) != 1 || fsh.Dims(2) != 1 || ish.Dims(1) != b->H || ish.Dims(2) != b->W ||
          ish.Dims(3) != b->N || osh.Dims(3) != b->M)
        Fatal("expansion conv does not match the plan", idx);
      g_blk = b; g_blk_no = no;
      StashConv(&g_e, params, mult, shift, ish, in, filt, bias, osh, out);
      g_state = kGotE;
      return true;                                             // captured, nothing computed
    }
    case kGotD: {
      if (idx != g_blk->p) Fatal("unexpected convolution inside a fused block", idx);
      if (fsh.Dims(1) != 1 || fsh.Dims(2) != 1 || ish.Dims(3) != g_blk->M || osh.Dims(3) != g_blk->P)
        Fatal("projection conv does not match the plan", idx);
      StashConv(&g_p, params, mult, shift, ish, in, filt, bias, osh, out);
      RunBlock();
      g_state = kIdle; g_blk = nullptr;
      return true;
    }
    default:
      Fatal("convolution seen while a fused block was incomplete", idx);
  }
  return false;
}

bool OnDepthwise(const DepthwiseParams& params, const int32_t* mult, const int32_t* shift,
                 const RuntimeShape& ish, const int8_t* in, const RuntimeShape& fsh, const int8_t* filt,
                 const RuntimeShape& bsh, const int32_t* bias, const RuntimeShape& osh, int8_t* out) {
  (void)bsh; (void)ish; (void)in;
  const int idx = g_idx++;
  if (!g_enabled || g_state == kIdle) return false;            // e.g. the first block's depthwise (t=1)
  const bool pad_ok = (g_blk->pad >= 0) ? (g_state == kGotPad) : (g_state == kGotE);
  if (!pad_ok || idx != g_blk->d) Fatal("depthwise conv out of sequence", idx);
  if (fsh.Dims(1) != 3 || fsh.Dims(2) != 3 || params.depth_multiplier != 1 ||
      params.stride_height != g_blk->stride || params.stride_width != g_blk->stride ||
      osh.Dims(3) != g_blk->M || osh.Dims(1) != g_blk->oh || osh.Dims(2) != g_blk->ow)
    Fatal("depthwise conv does not match the plan", idx);
  g_d.filt = filt; g_d.bias = bias; g_d.mult = mult; g_d.shift = shift; g_d.out = out;
  g_d.stride_h = params.stride_height; g_d.stride_w = params.stride_width;
  g_d.pad_h = params.padding_values.height; g_d.pad_w = params.padding_values.width;
  g_d.out_h = osh.Dims(1); g_d.out_w = osh.Dims(2); g_d.out_c = osh.Dims(3);
  g_d.input_offset = params.input_offset; g_d.output_offset = params.output_offset;
  g_d.act_min = params.quantized_activation_min; g_d.act_max = params.quantized_activation_max;
  g_state = kGotD;
  return true;
}

bool OnPad() {
  const int idx = g_idx++;
  if (!g_enabled || g_state == kIdle) return false;
  if (g_state != kGotE || g_blk->pad != idx) Fatal("PAD out of sequence", idx);
  g_state = kGotPad;
  return true;
}

void GetTotals(int* blocks_run, uint32_t* hw_cycles, uint32_t* cpu_cycles) {
  int ran = 0; uint32_t hw = 0;
  for (int i = 0; i < kNumFusedBlocks; i++) if (g_stat[i].ran) { ran++; hw += g_stat[i].hw_cycles; }
  *blocks_run = ran; *hw_cycles = hw; *cpu_cycles = g_total_cpu;
}

void PrintReport() {
  printf("\n========================================\n");
  printf(" Fused DSC CFU statistics\n");
  printf("========================================\n");
  int ran = 0;
  uint32_t hw = 0, jobs = 0, strips = 0;
  for (int i = 0; i < kNumFusedBlocks; i++) if (g_stat[i].ran) { ran++; hw += g_stat[i].hw_cycles; jobs += g_stat[i].jobs; strips += g_stat[i].strips; }
  printf("Blocks run on CFU     : %d / %d\n", ran, kNumFusedBlocks);
  printf("CFU jobs (START)      : %lu\n", (unsigned long)jobs);
  printf("Row strips            : %lu\n", (unsigned long)strips);
  printf("CFU busy cycles       : %lu   (hardware counter, compute only)\n", (unsigned long)hw);
  printf("Fused-path CPU cycles : %lu   (weights + input load + compute + read-back)\n", (unsigned long)g_total_cpu);
  printf("\nblock  input        ->  out        hwcyc    cpucyc   strips\n");
  for (int i = 0; i < kNumFusedBlocks; i++) {
    const fused_block_t& b = kFusedBlocks[i];
    printf("%3d    %3dx%3dx%-3d -> %3dx%3dx%-3d  %8lu %9lu  %d\n", i, b.H, b.W, b.N, b.oh, b.ow, b.P,
           (unsigned long)g_stat[i].hw_cycles, (unsigned long)g_stat[i].cpu_cycles, (int)g_stat[i].strips);
  }
#ifdef FUSED_VERIFY
  printf("\nVerification vs C reference: %d blocks checked, %d mismatching bytes -> %s\n", g_verify_blocks,
         g_verify_bad, g_verify_bad ? "FAIL" : "PASS");
#endif
  printf("========================================\n");
}

}  // namespace fused

// Replacement for the (now non-inline) int8 depthwise kernel; see integer_ops/depthwise_conv.h
namespace reference_integer_ops {
void DepthwiseConvPerChannel(const DepthwiseParams& params, const int32_t* output_multiplier,
                             const int32_t* output_shift, const RuntimeShape& input_shape,
                             const int8_t* input_data, const RuntimeShape& filter_shape,
                             const int8_t* filter_data, const RuntimeShape& bias_shape,
                             const int32_t* bias_data, const RuntimeShape& output_shape,
                             int8_t* output_data) {
  if (fused::OnDepthwise(params, output_multiplier, output_shift, input_shape, input_data, filter_shape,
                         filter_data, bias_shape, bias_data, output_shape, output_data))
    return;
  DepthwiseConvPerChannelRef(params, output_multiplier, output_shift, input_shape, input_data, filter_shape,
                             filter_data, bias_shape, bias_data, output_shape, output_data);
}
}  // namespace reference_integer_ops
}  // namespace tflite
