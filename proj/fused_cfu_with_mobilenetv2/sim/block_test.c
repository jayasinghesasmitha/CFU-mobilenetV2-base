// block_test.c -- two-pass RTL test of dsc_run_block():
//   ./block_test log   <dir> [slots words group]   : run the real driver, log every CFU instruction
//   ./block_test check <dir> [slots words group]   : run it again feeding the RTL's responses, compare to golden
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "test_vectors.h"

static FILE *g_f; static int g_check;
uint32_t dsc_cfu_stub(int f7, uint32_t a, uint32_t b) {
  if (!g_check) { fprintf(g_f, "%x %08x %08x\n", f7, a, b); return 0; }
  unsigned v = 0; if (fscanf(g_f, "%x", &v) != 1) { fprintf(stderr, "response file too short\n"); exit(2); }
  return v;
}
int main(int argc, char **argv) {
  const dsc_layer_t *L = &dsc_tv_layer;
  g_check = !strcmp(argv[1], "check");
  char path[512]; snprintf(path, sizeof path, "%s/%s", argv[2], g_check ? "resp.txt" : "cmds.txt");
  g_f = fopen(path, g_check ? "r" : "w");
  if (argc >= 6) dsc_set_limits(atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
  const int n = L->out_h * L->out_w * L->P;
  int8_t *out = (int8_t *)calloc(n + 8, 1);
  int rc = dsc_run_block(L, dsc_tv_in, out);
  fclose(g_f);
  if (!g_check) { printf("[log] rc=%d blocks=%u strips=%u jobs=%u\n", rc, dsc_stats.blocks, dsc_stats.strips, dsc_stats.jobs); return rc; }
  int bad = 0; for (int i = 0; i < n; i++) bad += out[i] != dsc_tv_expected[i];
  printf("[check] rc=%d strips=%u jobs=%u hw_cycles(sum)=%u  mismatches=%d/%d  -> %s\n",
         rc, dsc_stats.strips, dsc_stats.jobs, dsc_stats.hw_cycles, bad, n, bad ? "FAIL" : "PASS");
  return bad != 0;
}
