#ifndef PROJ_MNV2_CFU_DEPTHWISE_MNV2_CFU_H_
#define PROJ_MNV2_CFU_DEPTHWISE_MNV2_CFU_H_

#include <stdint.h>

#include "cfu.h"

#ifdef __cplusplus
extern "C" {
#endif


// ============================================================
// CFU COMMANDS
// ============================================================
//
// IMPORTANT:
//
// These MUST be preprocessor constants.
//
// Do NOT change these to an enum.
//
// cfu.h -> riscv.h eventually stringifies the funct7 argument
// into an assembler ".word" expression.
//
// Therefore:
//
//     #define MNV2_FUSED_RESET 0
//
// is correct.
//
// An enum can result in the assembler seeing:
//
//     MNV2_FUSED_RESET
//
// instead of:
//
//     0
//
// which produces:
//
//     invalid operands (*UND* and *ABS* sections) for `<<'
// ============================================================


#define MNV2_FUSED_RESET             0
#define MNV2_FUSED_CONFIG            1
#define MNV2_FUSED_SET_SHIFTS        2

#define MNV2_FUSED_PUSH_INPUT        3

#define MNV2_FUSED_PUSH_EXP_WEIGHT   4
#define MNV2_FUSED_PUSH_EXP_BIAS     5

#define MNV2_FUSED_PUSH_DW_WEIGHT    6
#define MNV2_FUSED_PUSH_DW_BIAS      7

#define MNV2_FUSED_PUSH_PROJ_WEIGHT  8
#define MNV2_FUSED_PUSH_PROJ_BIAS    9

#define MNV2_FUSED_RUN              10
#define MNV2_FUSED_POP_OUTPUT       11
#define MNV2_FUSED_STATUS           12


// ============================================================
// Hardware limits
// ============================================================

#define MNV2_FUSED_MAX_INPUT_CHANNELS     8
#define MNV2_FUSED_MAX_EXPANDED_CHANNELS  8
#define MNV2_FUSED_MAX_OUTPUT_CHANNELS    8

#define MNV2_FUSED_KERNEL_TAPS            9


// ============================================================
// CFU operation helper
// ============================================================

#define MNV2_FUSED_OP(command, value) \
  cfu_op0((command), (uint32_t)(value), 0)


// ============================================================
// CONFIG packing
// ============================================================
//
// bits  7:0   = input channels
// bits 15:8   = expanded channels
// bits 23:16  = output channels
// ============================================================

#define MNV2_FUSED_PACK_CONFIG(cin, cexp, cout) \
  ( \
      (uint32_t)(cin) | \
      ((uint32_t)(cexp) << 8) | \
      ((uint32_t)(cout) << 16) \
  )


// ============================================================
// SHIFT packing
// ============================================================
//
// bits  7:0   = expansion shift
// bits 15:8   = depthwise shift
// bits 23:16  = projection shift
// ============================================================

#define MNV2_FUSED_PACK_SHIFTS(exp, dw, proj) \
  ( \
      (uint32_t)(exp) | \
      ((uint32_t)(dw) << 8) | \
      ((uint32_t)(proj) << 16) \
  )


#ifdef __cplusplus
}
#endif

#endif  // PROJ_MNV2_CFU_DEPTHWISE_MNV2_CFU_H_