// dsc_driver.c -- loads config/weights/feature maps into the fused DSC CFU and runs it.
//
//  * dsc_run_block()   : whole inverted-residual block (Ex -> Dw -> Pr). Handles blocks that do not
//                        fit the on-chip buffers by (a) processing the output in ROW STRIPS (the input
//                        strip includes its 3x3 halo; rows outside the image are padding) and
//                        (b) splitting P > 56 output channels into GROUPS (expansion+depthwise are
//                        recomputed per group).
#include "dsc_driver.h"

#ifdef DSC_CFU_STUB
  extern uint32_t dsc_cfu_stub(int f7, uint32_t a, uint32_t b);
  #define DSC_CFU(f7, a, b) dsc_cfu_stub((f7), (uint32_t)(a), (uint32_t)(b))
#else
  #include "cfu.h"
  #define DSC_CFU(f7, a, b) ((uint32_t)cfu_op0((f7), (uint32_t)(a), (uint32_t)(b)))
#endif

// NOTE: these must be #defines (not enums): cfu_op0() pastes them into assembler text.
#define F_PING   0
#define F_CFG    1
#define F_MEMW   2
#define F_START  3
#define F_STATUS 4
#define F_OUTRD  5
#define F_WAIT   6
#define F_CYCLES 7
#define F_SUMRD  8
enum { MEM_EXW = 0, MEM_EXB = 1, MEM_EXM = 2, MEM_EXS = 3, MEM_DWW = 4 /*..12*/,
       MEM_DWB = 13, MEM_DWM = 14, MEM_DWS = 15, MEM_PRW = 16, MEM_PRB = 17, MEM_PRM = 18,
       MEM_PRS = 19, MEM_IFM = 20, MEM_LUT1 = 21, MEM_LUT2 = 22, MEM_LUT3 = 23, MEM_GATE = 24 };
enum { C_H, C_W, C_N, C_M, C_P, C_STRIDE, C_OUT_H, C_OUT_W, C_ROWOFF, C_COLOFF,
       C_EX_INOFF, C_EX_ZP, C_EX_MIN, C_EX_MAX, C_DW_INOFF, C_DW_ZP, C_DW_MIN, C_DW_MAX,
       C_PR_INOFF, C_PR_ZP, C_PR_MIN, C_PR_MAX, C_WQN, C_NC, C_K, C_LUTEN,
       C_MODE, C_GIN1, C_GIN2, C_GOUTZP, C_GMULT, C_GSHIFT, C_GMIN, C_GMAX };

// ---- hardware limits (must match cfu.v parameters) ------------------------------------------
#define HW_NPE        56        // projection engines
#define HW_IF_SLOTS   1024      // 64-bit words per IFMAP bank        (IF_AW = 10)
#define HW_OUT_WORDS  4096      // 32-bit words in the output buffer  (OUT_AW = 12)
#define HW_EXW_BYTES  65536     // expansion weights M*N              (EXW_AW = 13 -> 8192 x 8B)
#define HW_MAX_M      1024

dsc_stats_t dsc_stats;
static int g_max_slots = HW_IF_SLOTS, g_max_words = HW_OUT_WORDS, g_group = HW_NPE;
static int g_gap = 0;

void dsc_set_limits(int s, int w, int g) { g_max_slots = s; g_max_words = w; g_group = g; }
void dsc_set_gap(int n) { g_gap = n; }
void dsc_reset_stats(void) { dsc_stats.hw_cycles = dsc_stats.jobs = dsc_stats.strips = dsc_stats.blocks = 0; }

static inline void cfg(int idx, int32_t v) { DSC_CFU(F_CFG, idx, v); }
static inline void memw(int id, uint32_t addr, uint32_t d) {
  DSC_CFU(F_MEMW, ((uint32_t)id << 27) | addr, d);
  for (int i = 0; i < g_gap; i++) DSC_CFU(F_PING, 0, 0);
}
typedef uint32_t __attribute__((may_alias)) u32a;
static inline uint32_t pack4(const int8_t *p) {
  return (uint32_t)(uint8_t)p[0] | ((uint32_t)(uint8_t)p[1] << 8) |
         ((uint32_t)(uint8_t)p[2] << 16) | ((uint32_t)(uint8_t)p[3] << 24);
}
static inline uint32_t ld4(const int8_t *p) {            // aligned fast path, byte fallback
  return (((uintptr_t)p & 3u) == 0) ? *(const u32a *)(const void *)p : pack4(p);
}

uint32_t dsc_ping(void) { return DSC_CFU(F_PING, 0, 0); }
void dsc_start(void)  { DSC_CFU(F_START, 0, 0); }
void dsc_wait(void)   { DSC_CFU(F_WAIT, 0, 0); }
void dsc_wait_poll(void) { while (DSC_CFU(F_STATUS, 0, 0) & 1u) { } }
// Renode aborts any single CFU instruction that runs too long, so by default we poll STATUS.
// Define DSC_WAIT_BLOCKING to use the blocking WAIT instruction (used by the RTL replay tests).
void dsc_run(void) {
  dsc_start();
#ifdef DSC_WAIT_BLOCKING
  dsc_wait();
#else
  dsc_wait_poll();
#endif
}
uint32_t dsc_hw_cycles(void) { return DSC_CFU(F_CYCLES, 0, 0); }

uint32_t dsc_predict_cycles(const dsc_layer_t *L) {
  uint32_t mn = (uint32_t)(L->M * L->N) / 8, pix = (uint32_t)(L->out_h * L->out_w);
  const uint32_t S = (dsc_kernel(L) == 5) ? 4u : 1u;           // sub-windows per output pixel
  mn *= S;
  uint32_t gap = (mn < (uint32_t)(L->P + 4)) ? (uint32_t)(L->P + 4) - mn : 0;
  return 43u + (uint32_t)L->P + pix * (2u * S + mn) + (pix - 1u) * gap;
}

// ---- loaders ---------------------------------------------------------------------------------
static void load_cfg_static(const dsc_layer_t *L) {
  const int NC = L->N / 8, WQ = (L->W + 2) / 3;
  cfg(C_W, L->W); cfg(C_N, L->N); cfg(C_M, L->M);
  cfg(C_STRIDE, L->stride); cfg(C_OUT_W, L->out_w); cfg(C_COLOFF, L->col_off);
  cfg(C_EX_INOFF, L->ex_in_off); cfg(C_EX_ZP, L->ex_out_zp); cfg(C_EX_MIN, L->ex_min); cfg(C_EX_MAX, L->ex_max);
  cfg(C_DW_INOFF, L->dw_in_off); cfg(C_DW_ZP, L->dw_out_zp); cfg(C_DW_MIN, L->dw_min); cfg(C_DW_MAX, L->dw_max);
  cfg(C_PR_INOFF, L->pr_in_off); cfg(C_PR_ZP, L->pr_out_zp); cfg(C_PR_MIN, L->pr_min); cfg(C_PR_MAX, L->pr_max);
  cfg(C_WQN, WQ * NC); cfg(C_NC, NC);
  cfg(C_K, dsc_kernel(L));
  cfg(C_LUTEN, (L->lut_f1 ? 1 : 0) | (L->lut_f2 ? 2 : 0) | (L->lut_out ? 4 : 0));
  if (L->lut_f1)  for (int i = 0; i < 256; i++) memw(MEM_LUT1, i, (uint8_t)L->lut_f1[i]);
  if (L->lut_f2)  for (int i = 0; i < 256; i++) memw(MEM_LUT2, i, (uint8_t)L->lut_f2[i]);
  if (L->lut_out) for (int i = 0; i < 256; i++) memw(MEM_LUT3, i, (uint8_t)L->lut_out[i]);
}

static void load_ex_dw(const dsc_layer_t *L) {
  for (int w = 0; w < L->M * L->N / 4; w++) memw(MEM_EXW, w, ld4(L->ex_w + 4 * w));
  for (int m = 0; m < L->M; m++) {
    memw(MEM_EXB, m, L->ex_b ? (uint32_t)L->ex_b[m] : 0u);
    memw(MEM_EXM, m, (uint32_t)L->ex_m[m]); memw(MEM_EXS, m, (uint32_t)L->ex_s[m]);
    memw(MEM_DWB, m, L->dw_b ? (uint32_t)L->dw_b[m] : 0u);
    memw(MEM_DWM, m, (uint32_t)L->dw_m[m]); memw(MEM_DWS, m, (uint32_t)L->dw_s[m]);
  }
  // Depthwise weights.  3x3: sub-window 0 only.  5x5: four 3x3 sub-windows s = 0..3 with origins
  // (0,0) (0,3) (3,0) (3,3); tap j of sub-window s is filter position (3*(s>>1) + j/3, 3*(s&1) + j%3),
  // positions outside the 5x5 kernel carry weight 0.  Bank j, address {s, channel>>2}.
  {
    const int K = dsc_kernel(L), nsub = (K == 5) ? 4 : 1;
    for (int sw = 0; sw < nsub; sw++)
      for (int j = 0; j < 9; j++) {
        const int ky = ((K == 5) ? 3 * (sw >> 1) : 0) + j / 3, kx = ((K == 5) ? 3 * (sw & 1) : 0) + j % 3;
        const int ok = (ky < K && kx < K);
        for (int w = 0; w < L->M / 4; w++)
          memw(MEM_DWW + j, ((uint32_t)sw << 8) | (uint32_t)w, ok ? ld4(L->dw_w + (ky * K + kx) * L->M + 4 * w) : 0u);
      }
  }
}

// projection filters p0 .. p0+pg-1 -> engines 0..pg-1
static void load_pr(const dsc_layer_t *L, int p0, int pg) {
  for (int p = 0; p < pg; p++) {
    for (int w = 0; w < L->M / 4; w++)
      memw(MEM_PRW, ((uint32_t)p << 8) | (uint32_t)w, ld4(L->pr_w + (p0 + p) * L->M + 4 * w));
    memw(MEM_PRB, p, L->pr_b ? (uint32_t)L->pr_b[p0 + p] : 0u);
    memw(MEM_PRM, p, (uint32_t)L->pr_m[p0 + p]); memw(MEM_PRS, p, (uint32_t)L->pr_s[p0 + p]);
  }
}

// rows [a, a+hs) of the NHWC input become local rows 0..hs-1 of the CFU input buffer
static void load_ifmap_rows(const dsc_layer_t *L, const int8_t *in, int a, int hs) {
  const int W = L->W, N = L->N, WQ = (W + 2) / 3, wpp = N / 4;
  for (int r = 0; r < hs; r++) {
    const int8_t *row = in + (size_t)(a + r) * W * N;
    const int bank_r = (r % 3) * 3, slot_r = (r / 3) * WQ;
    int cm = 0, cq = 0;
    for (int c = 0; c < W; c++) {
      const uint32_t base = ((uint32_t)(bank_r + cm) << 23);
      const int slot = (slot_r + cq) * wpp;
      const int8_t *px = row + c * N;
      for (int j = 0; j < wpp; j++) memw(MEM_IFM, base | (uint32_t)(slot + j), ld4(px + 4 * j));
      if (++cm == 3) { cm = 0; cq++; }
    }
  }
}

static void read_words(int8_t *dst_base, int out_w, int rows, int P, int p0, int pg, int oy0) {
  const int aligned = (((uintptr_t)dst_base & 3u) == 0);
  int w = 0;
  for (int py = 0; py < rows; py++)
    for (int px = 0; px < out_w; px++) {
      int8_t *dst = dst_base + ((size_t)(oy0 + py) * out_w + px) * P + p0;
      for (int k = 0; k < pg / 4; k++, w++) {
        uint32_t v = DSC_CFU(F_OUTRD, w, 0);
        if (aligned) *(u32a *)(void *)(dst + 4 * k) = v;
        else { dst[4*k] = (int8_t)v; dst[4*k+1] = (int8_t)(v >> 8); dst[4*k+2] = (int8_t)(v >> 16); dst[4*k+3] = (int8_t)(v >> 24); }
      }
    }
}

int dsc_block_supported(const dsc_layer_t *L) {
  if ((L->N % 8) || (L->M % 8) || (L->P % 4)) return 0;
  if (L->M > HW_MAX_M || L->M * L->N > HW_EXW_BYTES) return 0;
  if (L->H > 255 || L->W > 255 || L->H < 1 || L->W < 1) return 0;
  if (L->stride != 1 && L->stride != 2) return 0;
  if (L->K != 0 && L->K != 3 && L->K != 5) return 0;
  if (L->P < 1 || L->P % 4) return 0;
  if (L->N / 8 > 16) return 0;
  return 1;
}

int dsc_run_block(const dsc_layer_t *L, const int8_t *in, int8_t *out) {
  if (!dsc_block_supported(L)) return -1;
  const int H = L->H, W = L->W, P = L->P, s = L->stride, oh = L->out_h, ow = L->out_w;
  const int WQ = (W + 2) / 3, NC = L->N / 8, KK = dsc_kernel(L);
  const int gsz = (P < g_group) ? P : g_group;
  const int ngroups = (P + gsz - 1) / gsz;

  // largest number of output rows per strip that fits both the output buffer and the IFMAP banks
  int rows = oh;
  if (rows > g_max_words / (ow * gsz / 4)) rows = g_max_words / (ow * gsz / 4);
  while (rows > 1) {
    int hs = (rows - 1) * s + KK; if (hs > H) hs = H;
    if (((hs + 2) / 3) * WQ * NC <= g_max_slots) break;
    rows--;
  }
  { int hs = (rows - 1) * s + KK; if (hs > H) hs = H;
    if (rows < 1 || ((hs + 2) / 3) * WQ * NC > g_max_slots) return -2; }

  load_cfg_static(L);
  cfg(C_H, H);
  load_ex_dw(L);
  if (ngroups == 1) { load_pr(L, 0, P); cfg(C_P, P); }

  for (int oy0 = 0; oy0 < oh; oy0 += rows) {
    const int oy1 = (oy0 + rows < oh) ? oy0 + rows : oh;
    const int r_lo = oy0 * s + L->row_off, r_hi = (oy1 - 1) * s + L->row_off + (KK - 1);
    const int a = r_lo < 0 ? 0 : r_lo, b = r_hi > H - 1 ? H - 1 : r_hi, hs = b - a + 1;
    load_ifmap_rows(L, in, a, hs);
    cfg(C_H, hs); cfg(C_OUT_H, oy1 - oy0); cfg(C_ROWOFF, r_lo - a);
    for (int g = 0; g < ngroups; g++) {
      const int p0 = g * gsz, pg = (P - p0 < gsz) ? P - p0 : gsz;
      if (ngroups > 1) { load_pr(L, p0, pg); cfg(C_P, pg); }
      dsc_run();
      dsc_stats.hw_cycles += dsc_hw_cycles();
      dsc_stats.jobs++;
      read_words(out, ow, oy1 - oy0, P, p0, pg, oy0);
    }
    dsc_stats.strips++;
  }
  dsc_stats.blocks++;
  return 0;
}

// ---- legacy single-shot API (whole map resident; used by the self-test / RTL replay) -----------
void dsc_load_layer(const dsc_layer_t *L) {
  load_cfg_static(L); cfg(C_H, L->H); cfg(C_OUT_H, L->out_h); cfg(C_ROWOFF, L->row_off); cfg(C_P, L->P);
  load_ex_dw(L); load_pr(L, 0, L->P);
}
void dsc_load_ifmap(const dsc_layer_t *L, const int8_t *in) { load_ifmap_rows(L, in, 0, L->H); }
void dsc_read_output(const dsc_layer_t *L, int8_t *out) { read_words(out, L->out_w, L->out_h, L->P, 0, L->P, 0); }

// ======================================================================================================
// squeeze-and-excite
// ======================================================================================================
static int g_se_a, g_se_hs, g_se_rowoff;     // input rows kept resident between the two passes

static int se_rows_ok(const dsc_layer_t *L, int gsz) {
  const int WQ = (L->W + 2) / 3, NC = L->N / 8, KK = dsc_kernel(L), s = L->stride;
  int hs = (L->out_h - 1) * s + KK; if (hs > L->H) hs = L->H;
  return L->out_h * L->out_w * gsz / 4 <= g_max_words && ((hs + 2) / 3) * WQ * NC <= g_max_slots;
}

int dsc_se_supported(const dsc_layer_t *L) {
  if (!dsc_block_supported(L)) return 0;
  return se_rows_ok(L, (L->P < g_group) ? L->P : g_group);
}

int dsc_se_pass1(const dsc_layer_t *L, const int8_t *in, int32_t *sums) {
  if (!dsc_se_supported(L)) return -1;
  const int H = L->H, s = L->stride, KK = dsc_kernel(L);
  load_cfg_static(L);
  load_ex_dw(L);
  const int r_lo = L->row_off, r_hi = (L->out_h - 1) * s + L->row_off + (KK - 1);
  g_se_a = r_lo < 0 ? 0 : r_lo;
  const int b = r_hi > H - 1 ? H - 1 : r_hi;
  g_se_hs = b - g_se_a + 1; g_se_rowoff = r_lo - g_se_a;
  load_ifmap_rows(L, in, g_se_a, g_se_hs);
  cfg(C_H, g_se_hs); cfg(C_OUT_H, L->out_h); cfg(C_ROWOFF, g_se_rowoff);
  cfg(C_P, (L->P < g_group) ? L->P : g_group);
  cfg(C_MODE, 1);                                          // squeeze
  dsc_run();
  dsc_stats.hw_cycles += dsc_hw_cycles(); dsc_stats.jobs++;
  for (int m = 0; m < L->M; m++) sums[m] = (int32_t)DSC_CFU(F_SUMRD, m, 0);
  cfg(C_MODE, 0);
  return 0;
}

int dsc_se_pass2(const dsc_layer_t *L, int8_t *out) {
  if (!L->se_gate || !dsc_se_supported(L)) return -1;
  const int P = L->P, gsz = (P < g_group) ? P : g_group, ngroups = (P + gsz - 1) / gsz;
  cfg(C_PR_INOFF, L->pr_in_off); cfg(C_PR_ZP, L->pr_out_zp); cfg(C_PR_MIN, L->pr_min); cfg(C_PR_MAX, L->pr_max);
  cfg(C_GIN1, L->se_in1_off); cfg(C_GIN2, L->se_in2_off); cfg(C_GOUTZP, L->se_out_zp);
  cfg(C_GMULT, L->se_mult); cfg(C_GSHIFT, L->se_shift); cfg(C_GMIN, L->se_min); cfg(C_GMAX, L->se_max);
  for (int m = 0; m < L->M; m++) memw(MEM_GATE, m, (uint32_t)(uint8_t)L->se_gate[m]);
  cfg(C_H, g_se_hs); cfg(C_OUT_H, L->out_h); cfg(C_ROWOFF, g_se_rowoff);
  cfg(C_MODE, 2);                                          // excite
  for (int g = 0; g < ngroups; g++) {
    const int p0 = g * gsz, pg = (P - p0 < gsz) ? P - p0 : gsz;
    load_pr(L, p0, pg); cfg(C_P, pg);
    dsc_run();
    dsc_stats.hw_cycles += dsc_hw_cycles(); dsc_stats.jobs++;
    read_words(out, L->out_w, L->out_h, P, p0, pg, 0);
  }
  cfg(C_MODE, 0);
  dsc_stats.blocks++; dsc_stats.strips++;
  return 0;
}
