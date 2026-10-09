// dsc_driver.h -- software side of the fused DSC CFU (Expansion -> Depthwise -> Projection).
#pragma once
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int H, W, N, M, P;                 // input HxWxN, expanded channels M, output channels P
  int stride, out_h, out_w;
  int row_off, col_off;              // = -pad_top, -pad_left of the depthwise 3x3
  int ex_in_off, ex_out_zp, ex_min, ex_max;   // in_off = -input_zero_point
  int dw_in_off, dw_out_zp, dw_min, dw_max;
  int pr_in_off, pr_out_zp, pr_min, pr_max;
  const int8_t  *ex_w;  const int32_t *ex_b, *ex_m, *ex_s;   // [M][N]   TFLite filter layout
  const int8_t  *dw_w;  const int32_t *dw_b, *dw_m, *dw_s;   // [3][3][M]
  const int8_t  *pr_w;  const int32_t *pr_b, *pr_m, *pr_s;   // [P][M]
  int K;                                                     // depthwise kernel: 3 or 5 (0 = 3)
  const int8_t *lut_f1, *lut_f2, *lut_out;                   // optional 256-entry tables (index = value+128), NULL = none
  // squeeze-and-excite: gate (int8 per expanded channel) multiplied onto F2 (after lut_f2) before the projection,
  // TFLite int8 MUL arithmetic: clamp(out_zp + mbqm((F2+in1_off)*(gate+in2_off), mult, shift), min, max)
  const int8_t *se_gate;                                     // NULL = no gate
  int se_in1_off, se_in2_off, se_out_zp, se_mult, se_shift, se_min, se_max;
} dsc_layer_t;

typedef struct {                     // accumulated over all dsc_run_block() calls
  uint32_t hw_cycles;                // sum of CFU busy cycles (hardware counter)
  uint32_t jobs;                     // START instructions issued
  uint32_t strips;                   // row strips processed
  uint32_t blocks;                   // blocks processed
} dsc_stats_t;
extern dsc_stats_t dsc_stats;

// ---- high level API used by the TFLM integration -----------------------------------------
// depthwise kernel size actually used (3 or 5)
static inline int dsc_kernel(const dsc_layer_t *L) { return L->K == 5 ? 5 : 3; }
int      dsc_block_supported(const dsc_layer_t *L);          // 1 if the block fits the hardware limits
int      dsc_run_block(const dsc_layer_t *L, const int8_t *in_nhwc, int8_t *out_nhwc);  // 0 = ok
void     dsc_reset_stats(void);

// ---- squeeze-and-excite blocks (two passes, E/DW weights and the input stay resident between them) ----------
int      dsc_se_supported(const dsc_layer_t *L);                // block fits in ONE row strip (P may need several groups)
// pass 1: run E->(table)->DW->(table) for every pixel and return sums[m] = sum over the map of F2[m] (M values)
int      dsc_se_pass1(const dsc_layer_t *L, const int8_t *in_nhwc, int32_t *sums);
// pass 2: recompute, multiply by L->se_gate, project; needs L's projection fields, writes the output tensor
int      dsc_se_pass2(const dsc_layer_t *L, int8_t *out_nhwc);
void     dsc_set_limits(int max_slots, int max_out_words, int group);  // test knob (small values force tiling)

// ---- low level API (single-shot, whole feature map in the CFU) -----------------------------
void     dsc_load_layer(const dsc_layer_t *L);
void     dsc_load_ifmap(const dsc_layer_t *L, const int8_t *in_nhwc);
void     dsc_start(void);
void     dsc_wait(void);
void     dsc_wait_poll(void);
void     dsc_run(void);
void     dsc_read_output(const dsc_layer_t *L, int8_t *out_nhwc);
uint32_t dsc_hw_cycles(void);
uint32_t dsc_ping(void);                                 // expect 0xD5C00001
void     dsc_set_gap(int n);
uint32_t dsc_predict_cycles(const dsc_layer_t *L);

// pure-C layer-by-layer reference (the software baseline)
void     dsc_ref_run(const dsc_layer_t *L, const int8_t *in, int8_t *f1_buf, int8_t *f2_buf, int8_t *out);

#ifdef __cplusplus
}
#endif
