#include "software_cfu.h"

#include <stdint.h>
#include <string.h>

#include "fused_depthwise.h"
#include "mnv2_cfu.h"


namespace {


struct State {

  int cin;
  int cexp;
  int cout;


  int shifts[3];


  int8_t input[
      9 * MNV2_FUSED_MAX_INPUT_CHANNELS];


  int8_t exp_weight[
      MNV2_FUSED_MAX_EXPANDED_CHANNELS *
      MNV2_FUSED_MAX_INPUT_CHANNELS];


  int32_t exp_bias[
      MNV2_FUSED_MAX_EXPANDED_CHANNELS];


  int8_t dw_weight[
      9 *
      MNV2_FUSED_MAX_EXPANDED_CHANNELS];


  int32_t dw_bias[
      MNV2_FUSED_MAX_EXPANDED_CHANNELS];


  int8_t proj_weight[
      MNV2_FUSED_MAX_OUTPUT_CHANNELS *
      MNV2_FUSED_MAX_EXPANDED_CHANNELS];


  int32_t proj_bias[
      MNV2_FUSED_MAX_OUTPUT_CHANNELS];


  int input_count;

  int exp_weight_count;
  int exp_bias_count;

  int dw_weight_count;
  int dw_bias_count;

  int proj_weight_count;
  int proj_bias_count;


  int8_t output[
      MNV2_FUSED_MAX_OUTPUT_CHANNELS];


  int output_count;


  bool configured;
  bool busy;
  bool result_valid;
  bool error;
};


State state;


// ============================================================
// RESET
// ============================================================

void Reset() {

  memset(
      &state,
      0,
      sizeof(state));
}


// ============================================================
// Check whether all parameters arrived
// ============================================================

bool Complete() {

  return
      state.configured &&

      state.input_count ==
          9 * state.cin &&

      state.exp_weight_count ==
          state.cexp *
          state.cin &&

      state.exp_bias_count ==
          state.cexp &&

      state.dw_weight_count ==
          9 *
          state.cexp &&

      state.dw_bias_count ==
          state.cexp &&

      state.proj_weight_count ==
          state.cout *
          state.cexp &&

      state.proj_bias_count ==
          state.cout;
}


// ============================================================
// STATUS
// ============================================================

uint32_t Status() {

  return
      (state.configured ? 1u : 0u) |

      (state.busy ?
           (1u << 1) :
           0u) |

      (state.result_valid ?
           (1u << 2) :
           0u) |

      (state.error ?
           (1u << 3) :
           0u) |

      (static_cast<uint32_t>(
           state.output_count)
       << 8) |

      (static_cast<uint32_t>(
           state.cout)
       << 16);
}


// ============================================================
// RUN software reference
// ============================================================

void Run() {

  Mnv2FusedParams p = {

      state.cin,
      state.cexp,
      state.cout,

      state.shifts[0],
      state.shifts[1],
      state.shifts[2],

      state.exp_weight,
      state.exp_bias,

      state.dw_weight,
      state.dw_bias,

      state.proj_weight,
      state.proj_bias
  };


  int8_t input_copy[
      9 *
      MNV2_FUSED_MAX_INPUT_CHANNELS];


  memcpy(
      input_copy,
      state.input,
      sizeof(input_copy));


  state.busy = true;


  state.result_valid =
      Mnv2FusedReference(
          p,
          input_copy,
          state.output);


  state.busy = false;


  state.output_count = 0;


  state.error =
      !state.result_valid;
}

}  // namespace


// ============================================================
// Software CFU
// ============================================================

extern "C"
uint32_t software_cfu(
    int funct3,
    int funct7,
    uint32_t in0,
    uint32_t in1) {

  (void)in1;


  if (funct3 != 0) {
    return 0;
  }


  switch (funct7) {


    // --------------------------------------------------------
    // RESET
    // --------------------------------------------------------

    case MNV2_FUSED_RESET:

      Reset();

      return 0;


    // --------------------------------------------------------
    // CONFIG
    // --------------------------------------------------------

    case MNV2_FUSED_CONFIG: {

      state.cin =
          in0 & 0xff;


      state.cexp =
          (in0 >> 8) & 0xff;


      state.cout =
          (in0 >> 16) & 0xff;


      state.configured =
          state.cin > 0 &&
          state.cin <=
              MNV2_FUSED_MAX_INPUT_CHANNELS &&

          state.cexp > 0 &&
          state.cexp <=
              MNV2_FUSED_MAX_EXPANDED_CHANNELS &&

          state.cout > 0 &&
          state.cout <=
              MNV2_FUSED_MAX_OUTPUT_CHANNELS;


      state.error =
          !state.configured;


      state.input_count = 0;

      state.exp_weight_count = 0;
      state.exp_bias_count = 0;

      state.dw_weight_count = 0;
      state.dw_bias_count = 0;

      state.proj_weight_count = 0;
      state.proj_bias_count = 0;

      state.output_count = 0;

      state.result_valid = false;


      return 0;
    }


    // --------------------------------------------------------
    // SHIFTS
    // --------------------------------------------------------

    case MNV2_FUSED_SET_SHIFTS:

      state.shifts[0] =
          in0 & 31;

      state.shifts[1] =
          (in0 >> 8) & 31;

      state.shifts[2] =
          (in0 >> 16) & 31;

      return 0;


    // --------------------------------------------------------
    // INPUT
    // --------------------------------------------------------

    case MNV2_FUSED_PUSH_INPUT:

      if (state.input_count <
          9 * state.cin) {

        state.input[
            state.input_count++] =
            static_cast<int8_t>(in0);

      } else {

        state.error = true;
      }

      return 0;


    // --------------------------------------------------------
    // EXPANSION WEIGHT
    // --------------------------------------------------------

    case MNV2_FUSED_PUSH_EXP_WEIGHT:

      if (state.exp_weight_count <
          state.cexp *
          state.cin) {

        state.exp_weight[
            state.exp_weight_count++] =
            static_cast<int8_t>(in0);

      } else {

        state.error = true;
      }

      return 0;


    // --------------------------------------------------------
    // EXPANSION BIAS
    // --------------------------------------------------------

    case MNV2_FUSED_PUSH_EXP_BIAS:

      if (state.exp_bias_count <
          state.cexp) {

        state.exp_bias[
            state.exp_bias_count++] =
            static_cast<int32_t>(in0);

      } else {

        state.error = true;
      }

      return 0;


    // --------------------------------------------------------
    // DEPTHWISE WEIGHT
    // --------------------------------------------------------

    case MNV2_FUSED_PUSH_DW_WEIGHT:

      if (state.dw_weight_count <
          9 * state.cexp) {

        state.dw_weight[
            state.dw_weight_count++] =
            static_cast<int8_t>(in0);

      } else {

        state.error = true;
      }

      return 0;


    // --------------------------------------------------------
    // DEPTHWISE BIAS
    // --------------------------------------------------------

    case MNV2_FUSED_PUSH_DW_BIAS:

      if (state.dw_bias_count <
          state.cexp) {

        state.dw_bias[
            state.dw_bias_count++] =
            static_cast<int32_t>(in0);

      } else {

        state.error = true;
      }

      return 0;


    // --------------------------------------------------------
    // PROJECTION WEIGHT
    // --------------------------------------------------------

    case MNV2_FUSED_PUSH_PROJ_WEIGHT:

      if (state.proj_weight_count <
          state.cout *
          state.cexp) {

        state.proj_weight[
            state.proj_weight_count++] =
            static_cast<int8_t>(in0);

      } else {

        state.error = true;
      }

      return 0;


    // --------------------------------------------------------
    // PROJECTION BIAS
    // --------------------------------------------------------

    case MNV2_FUSED_PUSH_PROJ_BIAS:

      if (state.proj_bias_count <
          state.cout) {

        state.proj_bias[
            state.proj_bias_count++] =
            static_cast<int32_t>(in0);

      } else {

        state.error = true;
      }

      return 0;


    // --------------------------------------------------------
    // RUN
    // --------------------------------------------------------

    case MNV2_FUSED_RUN:

      if (Complete()) {

        Run();

      } else {

        state.error = true;
      }

      return 0;


    // --------------------------------------------------------
    // POP OUTPUT
    // --------------------------------------------------------

    case MNV2_FUSED_POP_OUTPUT:

      if (state.result_valid &&
          state.output_count <
              state.cout) {

        return static_cast<uint8_t>(
            state.output[
                state.output_count++]);
      }


      state.error = true;

      return 0;


    // --------------------------------------------------------
    // STATUS
    // --------------------------------------------------------

    case MNV2_FUSED_STATUS:

      return Status();


    default:

      state.error = true;

      return 0;
  }
}