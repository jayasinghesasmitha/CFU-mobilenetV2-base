// dsc_ref.c -- layer-by-layer INT8 reference (same math as TFLM reference_integer_ops).
// This is the software baseline AND the fused block's golden model on target.
#include "dsc_driver.h"

static int32_t srdhm(int32_t a, int32_t b) {
  if (a == INT32_MIN && b == INT32_MIN) return INT32_MAX;
  int64_t ab = (int64_t)a * b;
  int32_t nudge = ab >= 0 ? (1 << 30) : (1 - (1 << 30));
  return (int32_t)((ab + nudge) / (1ll << 31));
}
static int32_t rdbp(int32_t x, int e) {
  int32_t mask = (int32_t)((1ll << e) - 1);
  int32_t rem = x & mask;
  int32_t thr = (mask >> 1) + (x < 0 ? 1 : 0);
  return (x >> e) + (rem > thr ? 1 : 0);
}
static int32_t mbqm(int32_t x, int32_t mult, int shift) {
  int left = shift > 0 ? shift : 0, right = shift > 0 ? 0 : -shift;
  return rdbp(srdhm((int32_t)((uint32_t)x << left), mult), right);
}
static int8_t q8(int32_t acc, int32_t m, int32_t s, int zp, int lo, int hi) {
  int32_t v = mbqm(acc, m, s) + zp;
  return (int8_t)(v < lo ? lo : (v > hi ? hi : v));
}

void dsc_ref_run(const dsc_layer_t *L, const int8_t *in, int8_t *f1, int8_t *f2, int8_t *out) {
  const int H = L->H, W = L->W, N = L->N, M = L->M, P = L->P, K = dsc_kernel(L);
  // 1) expansion: writes the full F1 feature map (H*W*M)
  for (int i = 0; i < H * W; i++)
    for (int m = 0; m < M; m++) {
      int32_t acc = L->ex_b[m];
      for (int n = 0; n < N; n++) acc += (in[i * N + n] + L->ex_in_off) * L->ex_w[m * N + n];
      int8_t v = q8(acc, L->ex_m[m], L->ex_s[m], L->ex_out_zp, L->ex_min, L->ex_max);
      f1[i * M + m] = L->lut_f1 ? L->lut_f1[v + 128] : v;
    }
  // 2) depthwise 3x3: reads F1, writes the full F2 feature map
  for (int oy = 0; oy < L->out_h; oy++)
    for (int ox = 0; ox < L->out_w; ox++)
      for (int m = 0; m < M; m++) {
        int32_t acc = L->dw_b[m];
        for (int ky = 0; ky < K; ky++)
          for (int kx = 0; kx < K; kx++) {
            int iy = oy * L->stride + L->row_off + ky, ix = ox * L->stride + L->col_off + kx;
            if (iy < 0 || iy >= H || ix < 0 || ix >= W) continue;
            acc += (f1[(iy * W + ix) * M + m] + L->dw_in_off) * L->dw_w[(ky * K + kx) * M + m];
          }
        { int8_t v = q8(acc, L->dw_m[m], L->dw_s[m], L->dw_out_zp, L->dw_min, L->dw_max);
          f2[(oy * L->out_w + ox) * M + m] = L->lut_f2 ? L->lut_f2[v + 128] : v; }
      }
  // 2b) squeeze-and-excite gate (TFLite int8 MUL), applied to F2 before the projection
  if (L->se_gate)
    for (int i = 0; i < L->out_h * L->out_w; i++)
      for (int m = 0; m < M; m++) {
        int32_t prod = (f2[i * M + m] + L->se_in1_off) * (L->se_gate[m] + L->se_in2_off);
        int32_t v = L->se_out_zp + mbqm(prod, L->se_mult, L->se_shift);
        f2[i * M + m] = (int8_t)(v < L->se_min ? L->se_min : (v > L->se_max ? L->se_max : v));
      }
  // 3) projection
  for (int i = 0; i < L->out_h * L->out_w; i++)
    for (int p = 0; p < P; p++) {
      int32_t acc = L->pr_b[p];
      for (int m = 0; m < M; m++) acc += (f2[i * M + m] + L->pr_in_off) * L->pr_w[p * M + m];
      { int8_t v = q8(acc, L->pr_m[p], L->pr_s[p], L->pr_out_zp, L->pr_min, L->pr_max);
        out[i * P + p] = L->lut_out ? L->lut_out[v + 128] : v; }
    }
}
