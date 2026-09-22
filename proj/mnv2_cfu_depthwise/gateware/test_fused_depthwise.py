#!/usr/bin/env python3
import unittest

from amaranth_cfu import CfuTestBase

from gateware.fused_depthwise import (
    CMD_CONFIG,
    CMD_POP_OUTPUT,
    CMD_PUSH_DW_BIAS,
    CMD_PUSH_DW_WEIGHT,
    CMD_PUSH_EXP_BIAS,
    CMD_PUSH_EXP_WEIGHT,
    CMD_PUSH_INPUT,
    CMD_PUSH_PROJ_BIAS,
    CMD_PUSH_PROJ_WEIGHT,
    CMD_RESET,
    CMD_RUN,
    CMD_SET_SHIFTS,
    CMD_STATUS,
    make_cfu,
)


def clamp_i8(value):
    return max(-128, min(127, value))


def reference(patch, exp_weight, exp_bias, dw_weight, dw_bias,
              proj_weight, proj_bias, shifts):
    exp_shift, dw_shift, proj_shift = shifts
    expanded = []
    for pixel in patch:
        expanded.append([
            clamp_i8((exp_bias[e] + sum(pixel[i] * exp_weight[e][i]
                                        for i in range(len(pixel)))) >> exp_shift)
            for e in range(len(exp_bias))
        ])
    depthwise = [
        clamp_i8((dw_bias[e] + sum(expanded[t][e] * dw_weight[t][e]
                                   for t in range(9))) >> dw_shift)
        for e in range(len(exp_bias))
    ]
    return [
        clamp_i8((proj_bias[o] + sum(depthwise[e] * proj_weight[o][e]
                                     for e in range(len(exp_bias)))) >> proj_shift)
        for o in range(len(proj_bias))
    ]


def op(command, value=0):
    return ((0, command, value, 0), 0)


class FusedDepthwiseCfuTest(CfuTestBase):
    def create_dut(self):
        return make_cfu()

    def test_fused_pixel_matches_reference(self):
        cin, cexp, cout = 2, 2, 2
        shifts = (1, 2, 1)
        patch = [[t - 4, 5 - t] for t in range(9)]
        exp_weight = [[2, -1], [-3, 2]]
        exp_bias = [3, -5]
        dw_weight = [[1 + (t % 3), -2 + (t % 2)] for t in range(9)]
        dw_bias = [7, -9]
        proj_weight = [[2, -1], [-1, 3]]
        proj_bias = [4, -6]
        expected = reference(patch, exp_weight, exp_bias, dw_weight, dw_bias,
                             proj_weight, proj_bias, shifts)

        stream = [
            op(CMD_RESET),
            op(CMD_CONFIG, cin | (cexp << 8) | (cout << 16)),
            op(CMD_SET_SHIFTS, shifts[0] | (shifts[1] << 8) | (shifts[2] << 16)),
        ]
        stream += [op(CMD_PUSH_INPUT, value) for pixel in patch for value in pixel]
        stream += [op(CMD_PUSH_EXP_WEIGHT, value)
                   for channel in exp_weight for value in channel]
        stream += [op(CMD_PUSH_EXP_BIAS, value) for value in exp_bias]
        stream += [op(CMD_PUSH_DW_WEIGHT, value)
                   for tap_weights in dw_weight for value in tap_weights]
        stream += [op(CMD_PUSH_DW_BIAS, value) for value in dw_bias]
        stream += [op(CMD_PUSH_PROJ_WEIGHT, value)
                   for channel in proj_weight for value in channel]
        stream += [op(CMD_PUSH_PROJ_BIAS, value) for value in proj_bias]
        stream += [op(CMD_RUN)]
        stream += [((0, CMD_POP_OUTPUT, 0, 0), value) for value in expected]
        self.run_ops(stream)

    def test_incomplete_stream_sets_sticky_error(self):
        cin, cexp, cout = 1, 1, 1
        self.run_ops([
            op(CMD_RESET),
            op(CMD_CONFIG, cin | (cexp << 8) | (cout << 16)),
            op(CMD_RUN),
            ((0, CMD_STATUS, 0, 0), 1 | (1 << 3) | (cout << 16)),
        ])

    def test_rejects_oversized_configuration(self):
        self.run_ops([
            op(CMD_RESET),
            op(CMD_CONFIG, 9 | (1 << 8) | (1 << 16)),
            ((0, CMD_STATUS, 0, 0), 1 << 3),
        ])


if __name__ == "__main__":
    unittest.main()
