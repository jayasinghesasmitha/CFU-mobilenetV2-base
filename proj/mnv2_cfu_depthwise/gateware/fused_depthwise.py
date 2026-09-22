#!/usr/bin/env python3
"""Small, CPU-fed fused Expansion -> 3x3 Depthwise -> Projection CFU.

The unit computes one output pixel per RUN. All tensors are explicitly pushed
through custom instructions; there is no DMA and no access to system memory.
"""

from amaranth import Array, Cat, Const, Mux, Repl, Signal, signed
from amaranth_cfu import InstructionBase, simple_cfu

MAX_INPUT_CHANNELS = 8
MAX_EXPANDED_CHANNELS = 8
MAX_OUTPUT_CHANNELS = 8
KERNEL_TAPS = 9

CMD_RESET = 0
CMD_CONFIG = 1
CMD_SET_SHIFTS = 2
CMD_PUSH_INPUT = 3
CMD_PUSH_EXP_WEIGHT = 4
CMD_PUSH_EXP_BIAS = 5
CMD_PUSH_DW_WEIGHT = 6
CMD_PUSH_DW_BIAS = 7
CMD_PUSH_PROJ_WEIGHT = 8
CMD_PUSH_PROJ_BIAS = 9
CMD_RUN = 10
CMD_POP_OUTPUT = 11
CMD_STATUS = 12


class FusedDepthwiseInstruction(InstructionBase):
    """Stateful command engine for a bounded fused inverted-residual kernel.

    Arithmetic at each stage is:
      y = clamp_int8(accumulator >> stage_shift)

    Inputs and weights are signed int8 in operand 0's low byte. Biases are
    signed int32. CONFIG packs Cin, Cexp and Cout into bits 7:0, 15:8, 23:16.
    SET_SHIFTS packs the three unsigned shifts in the same positions.
    """

    def elab(self, m):
        cin = Signal(range(MAX_INPUT_CHANNELS + 1))
        cexp = Signal(range(MAX_EXPANDED_CHANNELS + 1))
        cout = Signal(range(MAX_OUTPUT_CHANNELS + 1))
        exp_shift = Signal(5)
        dw_shift = Signal(5)
        proj_shift = Signal(5)

        configured = Signal()
        busy = Signal()
        result_valid = Signal()
        error = Signal()

        input_mem = Array(
            Signal(signed(8), name=f"input_{n}")
            for n in range(KERNEL_TAPS * MAX_INPUT_CHANNELS))
        exp_weight = Array(
            Signal(signed(8), name=f"exp_w_{n}")
            for n in range(MAX_EXPANDED_CHANNELS * MAX_INPUT_CHANNELS))
        exp_bias = Array(
            Signal(signed(32), name=f"exp_b_{n}")
            for n in range(MAX_EXPANDED_CHANNELS))
        dw_weight = Array(
            Signal(signed(8), name=f"dw_w_{n}")
            for n in range(KERNEL_TAPS * MAX_EXPANDED_CHANNELS))
        dw_bias = Array(
            Signal(signed(32), name=f"dw_b_{n}")
            for n in range(MAX_EXPANDED_CHANNELS))
        proj_weight = Array(
            Signal(signed(8), name=f"proj_w_{n}")
            for n in range(MAX_OUTPUT_CHANNELS * MAX_EXPANDED_CHANNELS))
        proj_bias = Array(
            Signal(signed(32), name=f"proj_b_{n}")
            for n in range(MAX_OUTPUT_CHANNELS))

        expanded = Array(
            Signal(signed(8), name=f"expanded_{n}")
            for n in range(KERNEL_TAPS * MAX_EXPANDED_CHANNELS))
        depthwise = Array(
            Signal(signed(8), name=f"depthwise_{n}")
            for n in range(MAX_EXPANDED_CHANNELS))
        outputs = Array(
            Signal(signed(8), name=f"output_{n}")
            for n in range(MAX_OUTPUT_CHANNELS))

        input_count = Signal(range(KERNEL_TAPS * MAX_INPUT_CHANNELS + 1))
        exp_weight_count = Signal(range(MAX_EXPANDED_CHANNELS * MAX_INPUT_CHANNELS + 1))
        exp_bias_count = Signal(range(MAX_EXPANDED_CHANNELS + 1))
        dw_weight_count = Signal(range(KERNEL_TAPS * MAX_EXPANDED_CHANNELS + 1))
        dw_bias_count = Signal(range(MAX_EXPANDED_CHANNELS + 1))
        proj_weight_count = Signal(range(MAX_OUTPUT_CHANNELS * MAX_EXPANDED_CHANNELS + 1))
        proj_bias_count = Signal(range(MAX_OUTPUT_CHANNELS + 1))
        pop_index = Signal(range(MAX_OUTPUT_CHANNELS + 1))

        tap = Signal(range(KERNEL_TAPS))
        in_channel = Signal(range(MAX_INPUT_CHANNELS))
        exp_channel = Signal(range(MAX_EXPANDED_CHANNELS))
        out_channel = Signal(range(MAX_OUTPUT_CHANNELS))
        accumulator = Signal(signed(32))

        product = Signal(signed(16))
        total = Signal(signed(32))
        shifted = Signal(signed(32))
        saturated = Signal(signed(8))

        status = Signal(32)
        current_output = Signal(signed(8))
        m.d.comb += [
            status.eq(
                configured
                | (busy << 1)
                | (result_valid << 2)
                | (error << 3)
                | (pop_index << 8)
                | (cout << 16)
            ),
            current_output.eq(outputs[pop_index[:3]]),
            self.done.eq(0),
            self.output.eq(0),
            product.eq(0),
            total.eq(accumulator + product),
            shifted.eq(total),
            saturated.eq(Mux(shifted > 127, 127,
                             Mux(shifted < -128, -128, shifted[:8]))),
        ]

        def clear_stream_state():
            return [
                input_count.eq(0),
                exp_weight_count.eq(0),
                exp_bias_count.eq(0),
                dw_weight_count.eq(0),
                dw_bias_count.eq(0),
                proj_weight_count.eq(0),
                proj_bias_count.eq(0),
                pop_index.eq(0),
                result_valid.eq(0),
            ]

        with m.FSM(reset="WAIT"):
            with m.State("WAIT"):
                m.d.sync += busy.eq(0)
                with m.If(self.start):
                    with m.Switch(self.funct7):
                        with m.Case(CMD_RESET):
                            m.d.sync += [
                                configured.eq(0),
                                error.eq(0),
                                cin.eq(0),
                                cexp.eq(0),
                                cout.eq(0),
                            ] + clear_stream_state()
                            m.d.comb += self.done.eq(1)

                        with m.Case(CMD_CONFIG):
                            valid_config = (
                                (self.in0[:8] > 0)
                                & (self.in0[:8] <= MAX_INPUT_CHANNELS)
                                & (self.in0[8:16] > 0)
                                & (self.in0[8:16] <= MAX_EXPANDED_CHANNELS)
                                & (self.in0[16:24] > 0)
                                & (self.in0[16:24] <= MAX_OUTPUT_CHANNELS)
                            )
                            m.d.sync += [
                                cin.eq(Mux(valid_config, self.in0[:8], 0)),
                                cexp.eq(Mux(valid_config, self.in0[8:16], 0)),
                                cout.eq(Mux(valid_config, self.in0[16:24], 0)),
                                configured.eq(valid_config),
                                error.eq(~valid_config),
                            ] + clear_stream_state()
                            m.d.comb += self.done.eq(1)

                        with m.Case(CMD_SET_SHIFTS):
                            valid_shifts = (
                                (self.in0[:8] <= 31)
                                & (self.in0[8:16] <= 31)
                                & (self.in0[16:24] <= 31)
                            )
                            m.d.sync += [
                                exp_shift.eq(self.in0[:5]),
                                dw_shift.eq(self.in0[8:13]),
                                proj_shift.eq(self.in0[16:21]),
                                error.eq(error | ~valid_shifts),
                            ]
                            m.d.comb += self.done.eq(1)

                        with m.Case(CMD_PUSH_INPUT):
                            with m.If(configured & (input_count < (KERNEL_TAPS * cin))):
                                m.d.sync += [
                                    input_mem[input_count].eq(self.in0[:8]),
                                    input_count.eq(input_count + 1),
                                ]
                            with m.Else():
                                m.d.sync += error.eq(1)
                            m.d.comb += self.done.eq(1)

                        with m.Case(CMD_PUSH_EXP_WEIGHT):
                            with m.If(configured & (exp_weight_count < (cexp * cin))):
                                m.d.sync += [
                                    exp_weight[exp_weight_count].eq(self.in0[:8]),
                                    exp_weight_count.eq(exp_weight_count + 1),
                                ]
                            with m.Else():
                                m.d.sync += error.eq(1)
                            m.d.comb += self.done.eq(1)

                        with m.Case(CMD_PUSH_EXP_BIAS):
                            with m.If(configured & (exp_bias_count < cexp)):
                                m.d.sync += [
                                    exp_bias[exp_bias_count].eq(self.in0),
                                    exp_bias_count.eq(exp_bias_count + 1),
                                ]
                            with m.Else():
                                m.d.sync += error.eq(1)
                            m.d.comb += self.done.eq(1)

                        with m.Case(CMD_PUSH_DW_WEIGHT):
                            with m.If(configured & (dw_weight_count < (KERNEL_TAPS * cexp))):
                                m.d.sync += [
                                    dw_weight[dw_weight_count].eq(self.in0[:8]),
                                    dw_weight_count.eq(dw_weight_count + 1),
                                ]
                            with m.Else():
                                m.d.sync += error.eq(1)
                            m.d.comb += self.done.eq(1)

                        with m.Case(CMD_PUSH_DW_BIAS):
                            with m.If(configured & (dw_bias_count < cexp)):
                                m.d.sync += [
                                    dw_bias[dw_bias_count].eq(self.in0),
                                    dw_bias_count.eq(dw_bias_count + 1),
                                ]
                            with m.Else():
                                m.d.sync += error.eq(1)
                            m.d.comb += self.done.eq(1)

                        with m.Case(CMD_PUSH_PROJ_WEIGHT):
                            with m.If(configured & (proj_weight_count < (cout * cexp))):
                                m.d.sync += [
                                    proj_weight[proj_weight_count].eq(self.in0[:8]),
                                    proj_weight_count.eq(proj_weight_count + 1),
                                ]
                            with m.Else():
                                m.d.sync += error.eq(1)
                            m.d.comb += self.done.eq(1)

                        with m.Case(CMD_PUSH_PROJ_BIAS):
                            with m.If(configured & (proj_bias_count < cout)):
                                m.d.sync += [
                                    proj_bias[proj_bias_count].eq(self.in0),
                                    proj_bias_count.eq(proj_bias_count + 1),
                                ]
                            with m.Else():
                                m.d.sync += error.eq(1)
                            m.d.comb += self.done.eq(1)

                        with m.Case(CMD_RUN):
                            all_loaded = (
                                configured
                                & (input_count == (KERNEL_TAPS * cin))
                                & (exp_weight_count == (cexp * cin))
                                & (exp_bias_count == cexp)
                                & (dw_weight_count == (KERNEL_TAPS * cexp))
                                & (dw_bias_count == cexp)
                                & (proj_weight_count == (cout * cexp))
                                & (proj_bias_count == cout)
                            )
                            with m.If(all_loaded):
                                m.d.sync += [
                                    busy.eq(1),
                                    result_valid.eq(0),
                                    tap.eq(0),
                                    exp_channel.eq(0),
                                    in_channel.eq(0),
                                    pop_index.eq(0),
                                ]
                                m.next = "EXP_INIT"
                            with m.Else():
                                m.d.sync += error.eq(1)
                                m.d.comb += self.done.eq(1)

                        with m.Case(CMD_POP_OUTPUT):
                            with m.If(result_valid & (pop_index < cout)):
                                m.d.comb += self.output.eq(Cat(
                                    current_output,
                                    Repl(current_output[7], 24)))
                                m.d.sync += pop_index.eq(pop_index + 1)
                                with m.If((pop_index + 1) == cout):
                                    m.d.sync += result_valid.eq(0)
                            with m.Else():
                                m.d.sync += error.eq(1)
                            m.d.comb += self.done.eq(1)

                        with m.Case(CMD_STATUS):
                            m.d.comb += [self.output.eq(status), self.done.eq(1)]

                        with m.Default():
                            m.d.sync += error.eq(1)
                            m.d.comb += self.done.eq(1)

            with m.State("EXP_INIT"):
                m.d.sync += [
                    accumulator.eq(exp_bias[exp_channel]),
                    in_channel.eq(0),
                ]
                m.next = "EXP_MAC"

            with m.State("EXP_MAC"):
                m.d.comb += [
                    product.eq(
                        input_mem[tap * cin + in_channel]
                        * exp_weight[exp_channel * cin + in_channel]),
                    shifted.eq(total >> exp_shift),
                ]
                with m.If((in_channel + 1) == cin):
                    m.d.sync += expanded[
                        tap * cexp + exp_channel].eq(saturated)
                    with m.If((exp_channel + 1) == cexp):
                        with m.If(tap == (KERNEL_TAPS - 1)):
                            m.d.sync += exp_channel.eq(0)
                            m.next = "DW_INIT"
                        with m.Else():
                            m.d.sync += [tap.eq(tap + 1), exp_channel.eq(0)]
                            m.next = "EXP_INIT"
                    with m.Else():
                        m.d.sync += exp_channel.eq(exp_channel + 1)
                        m.next = "EXP_INIT"
                with m.Else():
                    m.d.sync += [accumulator.eq(total), in_channel.eq(in_channel + 1)]

            with m.State("DW_INIT"):
                m.d.sync += [accumulator.eq(dw_bias[exp_channel]), tap.eq(0)]
                m.next = "DW_MAC"

            with m.State("DW_MAC"):
                m.d.comb += [
                    product.eq(
                        expanded[tap * cexp + exp_channel]
                        * dw_weight[tap * cexp + exp_channel]),
                    shifted.eq(total >> dw_shift),
                ]
                with m.If(tap == (KERNEL_TAPS - 1)):
                    m.d.sync += depthwise[exp_channel].eq(saturated)
                    with m.If((exp_channel + 1) == cexp):
                        m.d.sync += out_channel.eq(0)
                        m.next = "PROJ_INIT"
                    with m.Else():
                        m.d.sync += exp_channel.eq(exp_channel + 1)
                        m.next = "DW_INIT"
                with m.Else():
                    m.d.sync += [accumulator.eq(total), tap.eq(tap + 1)]

            with m.State("PROJ_INIT"):
                m.d.sync += [accumulator.eq(proj_bias[out_channel]), exp_channel.eq(0)]
                m.next = "PROJ_MAC"

            with m.State("PROJ_MAC"):
                m.d.comb += [
                    product.eq(
                        depthwise[exp_channel]
                        * proj_weight[out_channel * cexp + exp_channel]),
                    shifted.eq(total >> proj_shift),
                ]
                with m.If((exp_channel + 1) == cexp):
                    m.d.sync += outputs[out_channel].eq(saturated)
                    with m.If((out_channel + 1) == cout):
                        m.d.sync += [busy.eq(0), result_valid.eq(1), pop_index.eq(0)]
                        m.d.comb += self.done.eq(1)
                        m.next = "WAIT"
                    with m.Else():
                        m.d.sync += out_channel.eq(out_channel + 1)
                        m.next = "PROJ_INIT"
                with m.Else():
                    m.d.sync += [accumulator.eq(total), exp_channel.eq(exp_channel + 1)]


def make_cfu():
    return simple_cfu({0: FusedDepthwiseInstruction()})
