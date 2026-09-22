#ifndef PROJ_MNV2_CFU_DEPTHWISE_SOFTWARE_CFU_H_
#define PROJ_MNV2_CFU_DEPTHWISE_SOFTWARE_CFU_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

uint32_t software_cfu(
    int funct3,
    int funct7,
    uint32_t in0,
    uint32_t in1);

#ifdef __cplusplus
}
#endif

#endif  // PROJ_MNV2_CFU_DEPTHWISE_SOFTWARE_CFU_H_