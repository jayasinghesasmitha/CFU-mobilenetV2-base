"""Bit-accurate high-level model of the project's fused pixel dataflow.

This model deliberately allocates no expansion or depthwise feature maps.  For
each output pixel it forms only the 3x3 input patch, computes expansion values
for the nine taps, immediately reduces them in the depthwise stage, and feeds
those channel values directly into projection accumulators.
"""


def clamp_i8(value):
    return max(-128, min(127, value))


def shift_clamp(value, shift):
    return clamp_i8(value >> shift)


def fused_image(input_data, height, width, cin, expansion_weights,
                expansion_bias, depthwise_weights, depthwise_bias,
                projection_weights, projection_bias, shifts=(0, 0, 0),
                stride=1, padding=1, input_zero_point=0):
    """Runs zero-intermediate-buffer Ex->Dw->Pr fusion in NHWC order.

    Args:
      input_data: flat signed-int8 NHWC tensor for a single batch.
      expansion_weights: [cexp][cin].
      depthwise_weights: [9][cexp].
      projection_weights: [cout][cexp].
      stride: 1 or 2.
      padding: 0 (VALID) or 1 (SAME for 3x3).
      input_zero_point: value synthesized for out-of-bounds coordinates.
    """
    if stride not in (1, 2):
        raise ValueError("stride must be 1 or 2")
    if padding not in (0, 1):
        raise ValueError("padding must be 0 or 1")
    if len(input_data) != height * width * cin:
        raise ValueError("input size does not match shape")

    cexp = len(expansion_bias)
    cout = len(projection_bias)
    if len(expansion_weights) != cexp or any(len(w) != cin for w in expansion_weights):
        raise ValueError("invalid expansion weights")
    if len(depthwise_weights) != 9 or any(len(w) != cexp for w in depthwise_weights):
        raise ValueError("invalid depthwise weights")
    if len(projection_weights) != cout or any(len(w) != cexp for w in projection_weights):
        raise ValueError("invalid projection weights")

    out_h = (height + 2 * padding - 3) // stride + 1
    out_w = (width + 2 * padding - 3) // stride + 1
    if out_h <= 0 or out_w <= 0:
        return [], 0, 0

    exp_shift, dw_shift, proj_shift = shifts
    output = []
    for oy in range(out_h):
        for ox in range(out_w):
            depthwise_values = [0] * cexp
            for e in range(cexp):
                dw_acc = depthwise_bias[e]
                for ky in range(3):
                    iy = oy * stride + ky - padding
                    for kx in range(3):
                        ix = ox * stride + kx - padding
                        exp_acc = expansion_bias[e]
                        for i in range(cin):
                            if 0 <= iy < height and 0 <= ix < width:
                                value = input_data[(iy * width + ix) * cin + i]
                            else:
                                value = input_zero_point
                            exp_acc += value * expansion_weights[e][i]
                        expanded = shift_clamp(exp_acc, exp_shift)
                        dw_acc += expanded * depthwise_weights[ky * 3 + kx][e]
                depthwise_values[e] = shift_clamp(dw_acc, dw_shift)

            for o in range(cout):
                acc = projection_bias[o]
                for e in range(cexp):
                    acc += depthwise_values[e] * projection_weights[o][e]
                output.append(shift_clamp(acc, proj_shift))
    return output, out_h, out_w
