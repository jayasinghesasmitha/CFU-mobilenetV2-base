#!/usr/bin/env python3
"""
gen_vectors.py  --  golden model + test-vector generator for the fused DSC CFU.

Bit-exact model of TFLite/TFLM INT8 per-channel kernels:
   Expansion  : CONV_2D 1x1           (ReLU6 -> act range)
   Depthwise  : DEPTHWISE_CONV_2D 3x3 (SAME padding, stride 1 or 2, ReLU6)
   Projection : CONV_2D 1x1           (linear)

Outputs (in --outdir):
   test_vectors.h     C header with weights / params / input / expected output
   expected.hex       expected output as little-endian 32-bit words (one per line)
"""
import argparse, os
import numpy as np

# ---------------------------------------------------------------- TFLM math
def wrap32(x):
    x = np.asarray(x, dtype=np.int64)
    return ((x + 2**31) % 2**32) - 2**31

def srdhm(a, b):                      # SaturatingRoundingDoublingHighMul
    a = np.asarray(a, dtype=np.int64); b = np.asarray(b, dtype=np.int64)
    ab = a * b
    nudge = np.where(ab >= 0, 1 << 30, 1 - (1 << 30))
    t = ab + nudge
    q = np.where(t >= 0, t >> 31, -((-t) >> 31))      # C division truncates to 0
    ovf = (a == -2**31) & (b == -2**31)
    return np.where(ovf, 2**31 - 1, q)

def rdbp(x, e):                       # RoundingDivideByPOT
    x = np.asarray(x, dtype=np.int64); e = np.asarray(e, dtype=np.int64)
    mask = (np.int64(1) << e) - 1
    rem = x & mask
    thr = (mask >> 1) + (x < 0).astype(np.int64)
    return (x >> e) + (rem > thr).astype(np.int64)

def mbqm(x, mult, shift):             # MultiplyByQuantizedMultiplier
    shift = np.asarray(shift, dtype=np.int64)
    left = np.where(shift > 0, shift, 0)
    right = np.where(shift < 0, -shift, 0)
    xs = wrap32(np.asarray(x, dtype=np.int64) << left)
    return rdbp(srdhm(xs, mult), right)

def requant(acc, mult, shift, zp, lo, hi):
    return np.clip(mbqm(acc, mult, shift) + zp, lo, hi).astype(np.int8)

# ------------------------------------------------------------------ layers
def conv1x1(x, w, b, in_off, mult, shift, zp, lo, hi):
    acc = (x.astype(np.int64) + in_off) @ w.astype(np.int64).T + b
    return requant(acc, mult, shift, zp, lo, hi), acc

def dwconv3x3(f1, w, b, in_off, mult, shift, zp, lo, hi, stride, pt, pl, oh, ow):
    H, W, M = f1.shape
    fp = np.full((H + 8, W + 8, M), -in_off, dtype=np.int64)    # (v+in_off)==0 padding
    fp[4:4 + H, 4:4 + W] = f1
    acc = np.zeros((oh, ow, M), dtype=np.int64)
    ys0 = np.arange(oh) * stride - pt + 4
    xs0 = np.arange(ow) * stride - pl + 4
    for ky in range(3):
        for kx in range(3):
            win = fp[np.ix_(ys0 + ky, xs0 + kx)]
            acc += (win + in_off) * w[ky, kx].astype(np.int64)
    acc += b
    return requant(acc, mult, shift, zp, lo, hi), acc

def calib(acc, rng, target, positive_frac=0.0):
    """pick TFLite-style (multiplier in [2^30,2^31), shift) per channel"""
    M = acc.shape[-1]
    std = acc.reshape(-1, M).std(axis=0) + 1.0
    scale = target / std * rng.uniform(0.6, 1.4, M)
    npos = int(M * positive_frac)
    if npos:
        scale[:npos] = rng.uniform(1.2, 6.0, npos)                # force left shifts
    frac, e = np.frexp(scale)
    mult = np.round(frac * 2**31).astype(np.int64)
    over = mult >= 2**31
    mult[over] //= 2; e[over] += 1
    return mult.astype(np.int32), e.astype(np.int32)

def tflite_same_pad(inp, out, stride, k=3):
    total = max((out - 1) * stride + k - inp, 0)
    return total // 2

# -------------------------------------------------------------------- main
def emit_arr(f, ctype, name, a, per_line=16):
    a = np.asarray(a).reshape(-1)
    f.write(f"static const {ctype} {name}[{len(a)}] __attribute__((aligned(4))) = {{\n")
    for i in range(0, len(a), per_line):
        f.write("  " + ", ".join(str(int(v)) for v in a[i:i + per_line]) + ",\n")
    f.write("};\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--W", type=int, default=8)
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--P", type=int, default=8)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pos", type=float, default=0.0, help="fraction of channels using left shifts")
    ap.add_argument("--outdir", default="build")
    a = ap.parse_args()
    assert a.N % 8 == 0 and a.M % 8 == 0 and a.P % 4 == 0
    rng = np.random.default_rng(a.seed)
    H, W, N, M, P, s = a.H, a.W, a.N, a.M, a.P, a.stride
    OH, OW = -(-H // s), -(-W // s)
    pt, pl = tflite_same_pad(H, OH, s), tflite_same_pad(W, OW, s)

    x    = rng.integers(-128, 128, (H, W, N)).astype(np.int8)
    in_zp = int(rng.integers(-30, 30))
    ex_w = np.clip(rng.normal(0, 40, (M, N)).round(), -127, 127).astype(np.int8)
    dw_w = np.clip(rng.normal(0, 40, (3, 3, M)).round(), -127, 127).astype(np.int8)
    pr_w = np.clip(rng.normal(0, 40, (P, M)).round(), -127, 127).astype(np.int8)
    ex_b = rng.integers(-4000, 4000, M).astype(np.int32)
    dw_b = rng.integers(-4000, 4000, M).astype(np.int32)
    pr_b = rng.integers(-4000, 4000, P).astype(np.int32)

    ex_zp = int(rng.integers(-40, 40)); dw_zp = int(rng.integers(-40, 40)); pr_zp = int(rng.integers(-40, 40))
    ex_lo, ex_hi = int(max(-128, ex_zp)), int(min(127, ex_zp + rng.integers(40, 128)))
    dw_lo, dw_hi = int(max(-128, dw_zp)), int(min(127, dw_zp + rng.integers(40, 128)))
    pr_lo, pr_hi = -128, 127

    # stage 1: expansion (calibrate -> final)
    _, acc1 = conv1x1(x, ex_w, ex_b, -in_zp, np.ones(M, np.int32), np.zeros(M, np.int32), 0, -128, 127)
    ex_m, ex_s = calib(acc1, rng, 30.0, a.pos)
    f1, _ = conv1x1(x, ex_w, ex_b, -in_zp, ex_m, ex_s, ex_zp, ex_lo, ex_hi)
    # stage 2: depthwise
    _, acc2 = dwconv3x3(f1, dw_w, dw_b, -ex_zp, np.ones(M, np.int32), np.zeros(M, np.int32), 0, -128, 127, s, pt, pl, OH, OW)
    dw_m, dw_s = calib(acc2, rng, 30.0, a.pos)
    f2, _ = dwconv3x3(f1, dw_w, dw_b, -ex_zp, dw_m, dw_s, dw_zp, dw_lo, dw_hi, s, pt, pl, OH, OW)
    # stage 3: projection
    _, acc3 = conv1x1(f2, pr_w, pr_b, -dw_zp, np.ones(P, np.int32), np.zeros(P, np.int32), 0, -128, 127)
    pr_m, pr_s = calib(acc3, rng, 30.0, a.pos)
    out, _ = conv1x1(f2, pr_w, pr_b, -dw_zp, pr_m, pr_s, pr_zp, pr_lo, pr_hi)

    sat = np.mean((out == -128) | (out == 127))
    print(f"[gen] H={H} W={W} N={N} M={M} P={P} stride={s} -> {OH}x{OW}  pad_top={pt} pad_left={pl} "
          f"out saturation={sat*100:.1f}%  distinct out values={len(np.unique(out))}")

    os.makedirs(a.outdir, exist_ok=True)
    with open(os.path.join(a.outdir, "test_vectors.h"), "w") as f:
        f.write("// AUTO-GENERATED by scripts/gen_vectors.py -- do not edit\n#pragma once\n#include \"dsc_driver.h\"\n")
        emit_arr(f, "int8_t", "dsc_tv_in", x)
        emit_arr(f, "int8_t", "dsc_tv_ex_w", ex_w); emit_arr(f, "int32_t", "dsc_tv_ex_b", ex_b)
        emit_arr(f, "int32_t", "dsc_tv_ex_m", ex_m); emit_arr(f, "int32_t", "dsc_tv_ex_s", ex_s)
        emit_arr(f, "int8_t", "dsc_tv_dw_w", dw_w); emit_arr(f, "int32_t", "dsc_tv_dw_b", dw_b)
        emit_arr(f, "int32_t", "dsc_tv_dw_m", dw_m); emit_arr(f, "int32_t", "dsc_tv_dw_s", dw_s)
        emit_arr(f, "int8_t", "dsc_tv_pr_w", pr_w); emit_arr(f, "int32_t", "dsc_tv_pr_b", pr_b)
        emit_arr(f, "int32_t", "dsc_tv_pr_m", pr_m); emit_arr(f, "int32_t", "dsc_tv_pr_s", pr_s)
        emit_arr(f, "int8_t", "dsc_tv_expected", out)
        f.write(f"""
static const dsc_layer_t dsc_tv_layer = {{
  /*H,W,N,M,P*/ {H}, {W}, {N}, {M}, {P},
  /*stride,out_h,out_w*/ {s}, {OH}, {OW},
  /*row_off,col_off*/ {-pt}, {-pl},
  /*ex  in_off,out_zp,min,max*/ {-in_zp}, {ex_zp}, {ex_lo}, {ex_hi},
  /*dw  in_off,out_zp,min,max*/ {-ex_zp}, {dw_zp}, {dw_lo}, {dw_hi},
  /*pr  in_off,out_zp,min,max*/ {-dw_zp}, {pr_zp}, {pr_lo}, {pr_hi},
  dsc_tv_ex_w, dsc_tv_ex_b, dsc_tv_ex_m, dsc_tv_ex_s,
  dsc_tv_dw_w, dsc_tv_dw_b, dsc_tv_dw_m, dsc_tv_dw_s,
  dsc_tv_pr_w, dsc_tv_pr_b, dsc_tv_pr_m, dsc_tv_pr_s
}};
""")
    words = out.reshape(-1).view(np.uint8).reshape(-1, 4)
    with open(os.path.join(a.outdir, "expected.hex"), "w") as f:
        for w4 in words:
            f.write("%08x\n" % (int(w4[0]) | int(w4[1]) << 8 | int(w4[2]) << 16 | int(w4[3]) << 24))

if __name__ == "__main__":
    main()
