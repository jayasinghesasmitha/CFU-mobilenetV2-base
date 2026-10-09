// fused_dsc.cc -- see fused_dsc.h
#include "fused_dsc.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsc_driver.h"
#include "fused_plan.h"
#include "tensorflow/lite/kernels/internal/cppmath.h"
#include "tensorflow/lite/kernels/internal/max.h"
#include "tensorflow/lite/kernels/internal/min.h"
#include "tensorflow/lite/kernels/internal/reference/hard_swish.h"
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

struct ConvStash {
  const int8_t* in; const int8_t* filt; const int32_t* bias; const int32_t* mult; const int32_t* shift;
  int8_t* out;
  int in_h, in_w, in_c, out_h, out_w, out_c;
  int input_offset, output_offset, act_min, act_max;
};
struct DwStash {
  const int8_t* filt; const int32_t* bias; const int32_t* mult; const int32_t* shift; int8_t* out;
  int k, stride_h, stride_w, pad_h, pad_w, out_h, out_w, out_c;
  int input_offset, output_offset, act_min, act_max;
};
struct MulStash { const int8_t* gate; int in1_off, in2_off, out_zp, mult, shift, amin, amax; };
struct BlockStat { uint32_t cpu_cycles, hw_cycles, jobs, strips; int ran; int mismatches; };

constexpr int kMaxBlocks = 64;
const PlanBlock* g_plan = nullptr;     // nullptr -> generated plan
int g_plan_n = 0;
bool g_enabled = true;
int g_idx = 0;                         // ordinal of the next tracked operator
bool g_active = false;                 // a block is being captured
int g_next = 0;                        // ordinal the next captured operator must have
const PlanBlock* g_blk = nullptr;
int g_blk_no = -1;
ConvStash g_e, g_p;
DwStash g_d;
MulStash g_m;
int32_t g_sums[1024];
uint32_t g_hw0, g_jobs0, g_strips0, g_t0;
bool g_started = false;          // measurement of the current block has begun
int8_t g_lut1[256], g_lut2[256];
bool g_has_lut1 = false, g_has_lut2 = false;
BlockStat g_stat[kMaxBlocks];
uint32_t g_total_cpu = 0;
int g_verify_blocks = 0, g_verify_bad = 0;

// The projection output tensor may share arena memory with the (already dead) expansion input, so the fused
// result is first written to a private buffer when it overlaps the input.
static int8_t g_scratch_out[FUSED_MAX_OUT];
#ifdef FUSED_VERIFY
static int8_t g_vf1[FUSED_MAX_F1];
static int8_t g_vf2[FUSED_MAX_F2];
static int8_t g_vref[FUSED_MAX_OUT];
static int8_t g_vgate[1024];            // private copy of the gate (TFLM may place the output over the dead gate tensor)
static int8_t g_vin[FUSED_MAX_IN];     // private copy of the expansion input (TFLM may reuse its memory during SE)
#endif

int NumBlocks() { return g_plan ? g_plan_n : kNumFusedBlocks; }
PlanBlock Block(int i) {
  if (g_plan) return g_plan[i];
  const fused_block_t& b = kFusedBlocks[i];
  return PlanBlock{b.e, b.hs1, b.pad, b.d, b.hs2, b.mean, b.mulg, b.p, b.H, b.W, b.N, b.M, b.P, b.K, b.stride, b.oh, b.ow, b.pad_top, b.pad_left};
}
PlanBlock g_blk_copy;

[[noreturn]] void Fatal(const char* why, int idx) {
  printf("\nFUSED CFU PLAN MISMATCH at tracked op %d: %s\n", idx, why);
  printf("Regenerate src/fused_plan.h with tools/gen_fused_plan.py for this model.\n");
  abort();
}

bool FindByE(int idx, int* no) {
  for (int i = 0; i < NumBlocks(); i++)
    if (Block(i).e == idx) { *no = i; g_blk_copy = Block(i); g_blk = &g_blk_copy; return true; }
  return false;
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
  const PlanBlock& b = *g_blk;
  dsc_layer_t L;
  memset(&L, 0, sizeof(L));
  L.H = g_e.in_h; L.W = g_e.in_w; L.N = g_e.in_c; L.M = g_e.out_c; L.P = g_p.out_c;
  L.stride = g_d.stride_h; L.out_h = g_p.out_h; L.out_w = g_p.out_w; L.K = g_d.k;
  // window origin of the depthwise in UNPADDED coordinates: -(depthwise SAME padding) - (explicit PAD op top/left)
  L.row_off = -g_d.pad_h - b.pad_top; L.col_off = -g_d.pad_w - b.pad_left;
  L.ex_in_off = g_e.input_offset; L.ex_out_zp = g_e.output_offset; L.ex_min = g_e.act_min; L.ex_max = g_e.act_max;
  L.dw_in_off = g_d.input_offset; L.dw_out_zp = g_d.output_offset; L.dw_min = g_d.act_min; L.dw_max = g_d.act_max;
  L.pr_in_off = g_p.input_offset; L.pr_out_zp = g_p.output_offset; L.pr_min = g_p.act_min; L.pr_max = g_p.act_max;
  L.ex_w = g_e.filt; L.ex_b = g_e.bias; L.ex_m = g_e.mult; L.ex_s = g_e.shift;
  L.dw_w = g_d.filt; L.dw_b = g_d.bias; L.dw_m = g_d.mult; L.dw_s = g_d.shift;
  L.pr_w = g_p.filt; L.pr_b = g_p.bias; L.pr_m = g_p.mult; L.pr_s = g_p.shift;
  L.lut_f1 = g_has_lut1 ? g_lut1 : nullptr;
  L.lut_f2 = g_has_lut2 ? g_lut2 : nullptr;

  if (L.H != b.H || L.W != b.W || L.N != b.N || L.M != b.M || L.P != b.P || L.stride != b.stride || L.K != b.K ||
      L.out_h != b.oh || L.out_w != b.ow || g_d.out_h != b.oh || g_d.out_w != b.ow)
    Fatal("block dimensions differ from the plan", g_idx - 1);
  if (!dsc_block_supported(&L)) Fatal("block does not fit the CFU limits", g_idx - 1);

  if (b.mean >= 0) {
    L.se_gate = g_m.gate; L.se_in1_off = g_m.in1_off; L.se_in2_off = g_m.in2_off; L.se_out_zp = g_m.out_zp;
    L.se_mult = g_m.mult; L.se_shift = g_m.shift; L.se_min = g_m.amin; L.se_max = g_m.amax;
  }
  const int n_out = L.out_h * L.out_w * L.P;
  const int n_in = L.H * L.W * L.N;
  const bool overlap = (g_e.in < g_p.out + n_out) && (g_p.out < g_e.in + n_in);
  int8_t* dst = g_p.out;
  if (overlap) {
    if (n_out > (int)sizeof(g_scratch_out)) Fatal("output overlaps input and is too large for the scratch buffer", g_idx - 1);
    dst = g_scratch_out;
  }

  if (!g_started) { g_hw0 = dsc_stats.hw_cycles; g_jobs0 = dsc_stats.jobs; g_strips0 = dsc_stats.strips; g_t0 = fused_cycles(); }
  const int rc = (b.mean >= 0) ? dsc_se_pass2(&L, dst) : dsc_run_block(&L, g_e.in, dst);
  g_started = false;
  if (rc != 0) Fatal("fused block execution failed", g_idx - 1);

  BlockStat& s = g_stat[g_blk_no];
  s.hw_cycles = dsc_stats.hw_cycles - g_hw0; s.jobs = dsc_stats.jobs - g_jobs0;
  s.strips = dsc_stats.strips - g_strips0; s.ran = 1;

#ifdef FUSED_VERIFY
  if (n_out <= (int)sizeof(g_vref) && L.H * L.W * L.M <= (int)sizeof(g_vf1) && L.out_h * L.out_w * L.M <= (int)sizeof(g_vf2)) {
    dsc_layer_t Lv = L;
    if (b.mean >= 0) Lv.se_gate = g_vgate;
    dsc_ref_run(&Lv, (b.mean >= 0) ? g_vin : g_e.in, g_vf1, g_vf2, g_vref);
    int bad = 0;
    for (int i = 0; i < n_out; i++) bad += (g_vref[i] != dst[i]);
    s.mismatches = bad; g_verify_blocks++; g_verify_bad += bad;
  }
#endif
  if (overlap) memcpy(g_p.out, dst, n_out);
  s.cpu_cycles = fused_cycles() - g_t0;
  g_total_cpu += s.cpu_cycles;
}

// Operators inside an active block must arrive in consecutive order.
void Expect(int idx, int want, const char* what) {
  if (idx != want || g_next != idx) Fatal(what, idx);
  g_next++;
}

}  // namespace

void SetEnabled(bool e) { g_enabled = e; }
void SetPlanForTest(const PlanBlock* blocks, int n) { g_plan = blocks; g_plan_n = n; }

void ResetStats() {
  memset(g_stat, 0, sizeof(g_stat));
  g_total_cpu = 0; g_verify_blocks = 0; g_verify_bad = 0;
  g_idx = 0; g_active = false; g_blk = nullptr; g_started = false;     // a new inference starts
  dsc_reset_stats();
}

bool OnConv(const ConvParams& params, const int32_t* mult, const int32_t* shift, const RuntimeShape& ish,
            const int8_t* in, const RuntimeShape& fsh, const int8_t* filt, const RuntimeShape& bsh,
            const int32_t* bias, const RuntimeShape& osh, int8_t* out) {
  (void)bsh;
  const int idx = g_idx++;
  if (!g_enabled) return false;

  if (!g_active) {
    int no;
    if (!FindByE(idx, &no)) return false;                          // ordinary convolution
    const PlanBlock& b = *g_blk;
    if (fsh.Dims(1) != 1 || fsh.Dims(2) != 1 || ish.Dims(1) != b.H || ish.Dims(2) != b.W ||
        ish.Dims(3) != b.N || osh.Dims(3) != b.M)
      Fatal("expansion conv does not match the plan", idx);
    g_blk_no = no; g_has_lut1 = g_has_lut2 = false;
    StashConv(&g_e, params, mult, shift, ish, in, filt, bias, osh, out);
    g_active = true; g_next = idx + 1;
    return true;                                                   // captured, nothing computed
  }
  const PlanBlock& b = *g_blk;
  if (b.mean >= 0 && idx > b.mean && idx < b.mulg) return false;  // the two small SE convs run on the CPU
  Expect(idx, b.p, "unexpected convolution inside a fused block");
  if (fsh.Dims(1) != 1 || fsh.Dims(2) != 1 || ish.Dims(3) != b.M || osh.Dims(3) != b.P)
    Fatal("projection conv does not match the plan", idx);
  StashConv(&g_p, params, mult, shift, ish, in, filt, bias, osh, out);
  RunBlock();
  g_active = false; g_blk = nullptr;
  return true;
}

bool OnDepthwise(const DepthwiseParams& params, const int32_t* mult, const int32_t* shift,
                 const RuntimeShape& ish, const int8_t* in, const RuntimeShape& fsh, const int8_t* filt,
                 const RuntimeShape& bsh, const int32_t* bias, const RuntimeShape& osh, int8_t* out) {
  (void)bsh; (void)ish; (void)in;
  const int idx = g_idx++;
  if (!g_enabled || !g_active) return false;                       // e.g. the first block's depthwise (t=1)
  const PlanBlock& b = *g_blk;
  Expect(idx, b.d, "depthwise conv out of sequence");
  const int K = fsh.Dims(1);
  if ((K != 3 && K != 5) || fsh.Dims(2) != K || K != b.K || params.depth_multiplier != 1 ||
      params.stride_height != b.stride || params.stride_width != b.stride ||
      osh.Dims(3) != b.M || osh.Dims(1) != b.oh || osh.Dims(2) != b.ow)
    Fatal("depthwise conv does not match the plan", idx);
  g_d.filt = filt; g_d.bias = bias; g_d.mult = mult; g_d.shift = shift; g_d.out = out; g_d.k = K;
  g_d.stride_h = params.stride_height; g_d.stride_w = params.stride_width;
  g_d.pad_h = params.padding_values.height; g_d.pad_w = params.padding_values.width;
  g_d.out_h = osh.Dims(1); g_d.out_w = osh.Dims(2); g_d.out_c = osh.Dims(3);
  g_d.input_offset = params.input_offset; g_d.output_offset = params.output_offset;
  g_d.act_min = params.quantized_activation_min; g_d.act_max = params.quantized_activation_max;
  return true;
}

bool OnPad() {
  const int idx = g_idx++;
  if (!g_enabled || !g_active) return false;
  Expect(idx, g_blk->pad, "PAD out of sequence");
  return true;
}

// hard-swish: build the 256-entry table with TFLM's own reference function (bit-exact by construction)
bool OnHardSwish(const HardSwishParams& p) {
  const int idx = g_idx++;
  if (!g_enabled || !g_active) return false;
  const PlanBlock& b = *g_blk;
  int8_t* lut;
  if (idx == b.hs1) { lut = g_lut1; g_has_lut1 = true; }
  else if (idx == b.hs2) { lut = g_lut2; g_has_lut2 = true; }
  else Fatal("HARD_SWISH out of sequence", idx);
  Expect(idx, idx, "HARD_SWISH out of sequence");
  int8_t in[256];
  for (int i = 0; i < 256; i++) in[i] = (int8_t)(i - 128);
  RuntimeShape sh(1);
  sh.SetDim(0, 256);
  reference_ops::HardSwish<int8_t>(p, sh, in, sh, lut);
  return true;
}

// MEAN of a squeeze-and-excite block: pass 1 on the CFU, then TFLM's own arithmetic on the per-channel sums
bool OnMean(int in_zp, float in_scale, int out_zp, float out_scale, const int* in_dims, int8_t* out) {
  const int idx = g_idx++;
  if (!g_enabled || !g_active) return false;                       // e.g. the final global pooling
  const PlanBlock& b = *g_blk;
  if (b.mean < 0) Fatal("MEAN inside a block without squeeze-and-excite", idx);
  Expect(idx, b.mean, "MEAN out of sequence");
  if (in_dims[1] != b.oh || in_dims[2] != b.ow || in_dims[3] != b.M) Fatal("MEAN does not match the plan", idx);

  dsc_layer_t L;
  memset(&L, 0, sizeof(L));
  L.H = g_e.in_h; L.W = g_e.in_w; L.N = g_e.in_c; L.M = g_e.out_c; L.P = b.P;
  L.stride = g_d.stride_h; L.out_h = b.oh; L.out_w = b.ow; L.K = g_d.k;
  L.row_off = -g_d.pad_h - b.pad_top; L.col_off = -g_d.pad_w - b.pad_left;
  L.ex_in_off = g_e.input_offset; L.ex_out_zp = g_e.output_offset; L.ex_min = g_e.act_min; L.ex_max = g_e.act_max;
  L.dw_in_off = g_d.input_offset; L.dw_out_zp = g_d.output_offset; L.dw_min = g_d.act_min; L.dw_max = g_d.act_max;
  L.ex_w = g_e.filt; L.ex_b = g_e.bias; L.ex_m = g_e.mult; L.ex_s = g_e.shift;
  L.dw_w = g_d.filt; L.dw_b = g_d.bias; L.dw_m = g_d.mult; L.dw_s = g_d.shift;
  L.lut_f1 = g_has_lut1 ? g_lut1 : nullptr;
  L.lut_f2 = g_has_lut2 ? g_lut2 : nullptr;
  if (L.H != b.H || L.W != b.W || L.N != b.N || L.M != b.M || L.stride != b.stride || L.K != b.K ||
      g_d.out_h != b.oh || g_d.out_w != b.ow)
    Fatal("block dimensions differ from the plan", idx);

  g_hw0 = dsc_stats.hw_cycles; g_jobs0 = dsc_stats.jobs; g_strips0 = dsc_stats.strips; g_t0 = fused_cycles();
  g_started = true;
#ifdef FUSED_VERIFY
  if (L.H * L.W * L.N <= (int)sizeof(g_vin)) memcpy(g_vin, g_e.in, L.H * L.W * L.N);
#endif
  if (dsc_se_pass1(&L, g_e.in, g_sums) != 0) Fatal("squeeze pass failed", idx);

  // same arithmetic as reference_ops::Mean / QuantizedMeanOrSum (reduce.h) applied to the sums
  const int n = b.oh * b.ow;
  if (in_zp == out_zp && in_scale == out_scale) {
    for (int c = 0; c < b.M; c++) out[c] = static_cast<int8_t>(g_sums[c] / static_cast<int32_t>(n));
  } else {
    const float scale = in_scale / out_scale;
    const float bias = -in_zp * scale;
    for (int c = 0; c < b.M; c++) {
      float float_mean = static_cast<float>(g_sums[c]) / static_cast<float>(n);
      float result = TfLiteMin(TfLiteRound(float_mean * scale + bias) + out_zp, static_cast<float>(127));
      result = TfLiteMax(result, static_cast<float>(-128));
      out[c] = static_cast<int8_t>(result);
    }
  }
  g_next = b.mulg;                                                 // the two SE convs and the constant MUL follow
  return true;
}

bool OnMul(int in1_zp, int in2_zp, int out_zp, int32_t mult, int shift, int32_t amin, int32_t amax,
           const int* dims1, int nd1, const int8_t* d1, const int* dims2, int nd2, const int8_t* d2) {
  const int idx = g_idx++;
  if (!g_enabled || !g_active) return false;
  const PlanBlock& b = *g_blk;
  if (b.mean < 0) Fatal("MUL inside a block without squeeze-and-excite", idx);
  if (idx > b.mean && idx < b.mulg) return false;                  // the constant gate-scaling MUL runs on the CPU
  Expect(idx, b.mulg, "gating MUL out of sequence");
  // the gate is the operand with spatial size 1x1 ([1,1,1,M]), the other operand is the (never stored) F2 map
  const bool gate_is_2 = (nd2 == 4 && dims2[1] * dims2[2] == 1);
  const bool gate_is_1 = (nd1 == 4 && dims1[1] * dims1[2] == 1);
  if (gate_is_1 == gate_is_2) Fatal("cannot identify the gate operand of the MUL", idx);
  g_m.gate = gate_is_2 ? d2 : d1;
#ifdef FUSED_VERIFY
  memcpy(g_vgate, g_m.gate, b.M);
#endif
  g_m.in1_off = gate_is_2 ? -in1_zp : -in2_zp;                     // offset of the feature-map operand
  g_m.in2_off = gate_is_2 ? -in2_zp : -in1_zp;                     // offset of the gate operand
  g_m.out_zp = out_zp; g_m.mult = mult; g_m.shift = shift; g_m.amin = amin; g_m.amax = amax;
  return true;
}

void GetTotals(int* blocks_run, uint32_t* hw_cycles, uint32_t* cpu_cycles) {
  int ran = 0; uint32_t hw = 0;
  for (int i = 0; i < NumBlocks(); i++) if (g_stat[i].ran) { ran++; hw += g_stat[i].hw_cycles; }
  *blocks_run = ran; *hw_cycles = hw; *cpu_cycles = g_total_cpu;
}

void PrintReport() {
  printf("\n========================================\n");
  printf(" Fused DSC CFU statistics\n");
  printf("========================================\n");
  int ran = 0, k5 = 0, se = 0;
  uint32_t hw = 0, jobs = 0, strips = 0;
  for (int i = 0; i < NumBlocks(); i++) if (g_stat[i].ran) { ran++; hw += g_stat[i].hw_cycles; jobs += g_stat[i].jobs; strips += g_stat[i].strips; if (Block(i).K == 5) k5++; if (Block(i).mean >= 0) se++; }
  printf("Blocks run on CFU     : %d / %d   (%d with 5x5 depthwise, %d with squeeze-and-excite)\n", ran, NumBlocks(), k5, se);
  printf("CFU jobs (START)      : %lu\n", (unsigned long)jobs);
  printf("Row strips            : %lu\n", (unsigned long)strips);
  printf("CFU busy cycles       : %lu   (hardware counter, compute only)\n", (unsigned long)hw);
  printf("Fused-path CPU cycles : %lu   (weights + input load + compute + read-back)\n", (unsigned long)g_total_cpu);
  printf("\nblock  K SE  input        ->  out        hwcyc    cpucyc   strips\n");
  for (int i = 0; i < NumBlocks(); i++) {
    const PlanBlock b = Block(i);
    printf("%3d    %d %s   %3dx%3dx%-3d -> %3dx%3dx%-3d  %8lu %9lu  %d\n", i, b.K, b.mean >= 0 ? "SE" : "--", b.H, b.W, b.N, b.oh, b.ow, b.P,
           (unsigned long)g_stat[i].hw_cycles, (unsigned long)g_stat[i].cpu_cycles, (int)g_stat[i].strips);
#ifdef FUSED_VERIFY
    if (g_stat[i].mismatches) printf("       ^ block %d: %d bytes differ from the C reference\n", i, g_stat[i].mismatches);
#endif
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
