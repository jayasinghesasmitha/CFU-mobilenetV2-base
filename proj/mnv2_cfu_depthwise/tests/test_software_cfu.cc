#include <assert.h>
#include <stdint.h>

#include "mnv2_cfu.h"
#include "software_cfu.h"

int main() {
  assert(software_cfu(0, MNV2_FUSED_RESET, 0, 0) == 0);
  assert(software_cfu(0, MNV2_FUSED_CONFIG,
                     MNV2_FUSED_PACK_CONFIG(1, 1, 1), 0) == 0);
  assert((software_cfu(0, MNV2_FUSED_STATUS, 0, 0) & 1u) != 0);
  assert((software_cfu(0, MNV2_FUSED_RUN, 0, 0),
          (software_cfu(0, MNV2_FUSED_STATUS, 0, 0) & (1u << 3)) != 0));
  return 0;
}
