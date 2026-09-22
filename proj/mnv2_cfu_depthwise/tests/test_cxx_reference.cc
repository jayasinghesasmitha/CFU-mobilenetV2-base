#include <assert.h>
#include <stdint.h>

#include "fused_depthwise.h"

int main() {
  const int8_t input[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  const int8_t ew[1] = {2};
  const int32_t eb[1] = {0};
  const int8_t dw[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  const int32_t db[1] = {0};
  const int8_t pw[1] = {3};
  const int32_t pb[1] = {0};
  const Mnv2FusedParams params = {1, 1, 1, 0, 0, 0,
                                  ew, eb, dw, db, pw, pb};
  int8_t output[1] = {0};
  assert(Mnv2FusedReference(params, input, output));
  assert(output[0] == 127);
  return 0;
}
