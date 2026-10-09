#!/usr/bin/env python3
"""
gen_fused_plan.py -- finds every  CONV_2D(1x1) -> [HARD_SWISH] -> [PAD] -> DEPTHWISE_CONV_2D(3x3 | 5x5) -> [HARD_SWISH]
-> [SE: MEAN -> CONV_2D -> CONV_2D -> MUL -> MUL(gate)] -> CONV_2D(1x1)  inverted-residual block of a TFLite model
and writes src/fused_plan.h.

Operators are numbered in execution order, counting only CONV_2D, DEPTHWISE_CONV_2D, PAD, HARD_SWISH, MEAN and MUL
(the operators the fused-CFU hooks intercept).  The firmware counts the same way at run time.
Blocks that contain anything else between expansion and projection, or that do not fit the hardware limits
(see fits()), are NOT fused; they keep running on the CPU.

usage: python3 tools/gen_fused_plan.py model/<model>.tflite src/fused_plan.h [plan.txt]
(the optional plan.txt is read by the PC test harness: --plan plan.txt)
"""
import sys
import numpy as np
import tflite

TRACKED = ("CONV_2D", "DEPTHWISE_CONV_2D", "PAD", "HARD_SWISH", "MEAN", "MUL")


def load(path):
    buf = open(path, "rb").read()
    m = tflite.Model.GetRootAsModel(buf, 0)
    g = m.Subgraphs(0)
    names = {v: k for k, v in vars(tflite.BuiltinOperator).items() if not k.startswith("_")}
    codes = [max(m.OperatorCodes(i).BuiltinCode(), m.OperatorCodes(i).DeprecatedBuiltinCode())
             for i in range(m.OperatorCodesLength())]
    ops = []
    for i in range(g.OperatorsLength()):
        op = g.Operators(i)
        ops.append(dict(i=i, type=names.get(codes[op.OpcodeIndex()], "?"),
                        ins=[op.Inputs(j) for j in range(op.InputsLength())],
                        outs=[op.Outputs(j) for j in range(op.OutputsLength())], op=op))
    return m, g, ops


def fits(b):
    """mirror of the driver / hardware limits (src/dsc_driver.c: dsc_block_supported, dsc_se_supported)"""
    if b["N"] % 8 or b["M"] % 8 or b["P"] % 4: return False, "channels must be multiples of 8/8/4"
    if b["M"] > 1024 or b["M"] * b["N"] > 65536: return False, "M or M*N too large"
    if b["H"] > 255 or b["W"] > 255 or b["N"] // 8 > 16: return False, "H/W/N too large"
    if b["mean"] >= 0:                                    # squeeze-and-excite: must fit ONE row strip
        gsz = min(b["P"], 56); hs = min(b["H"], (b["oh"] - 1) * b["stride"] + b["K"])
        if b["oh"] * b["ow"] * gsz // 4 > 4096: return False, "output too large for one strip"
        if ((hs + 2) // 3) * ((b["W"] + 2) // 3) * (b["N"] // 8) > 1024: return False, "input too large for one strip"
    return True, ""


def main():
    m, g, ops = load(sys.argv[1])
    shape = lambda t: [int(x) for x in g.Tensors(t).ShapeAsNumpy()]
    zp = lambda t: int(g.Tensors(t).Quantization().ZeroPointAsNumpy()[0])
    users = {}
    for o in ops:
        for t in o["ins"]:
            users[t] = users.get(t, 0) + 1
    tracked = [o for o in ops if o["type"] in TRACKED]
    for k, o in enumerate(tracked):
        o["k"] = k

    def dw_options(o):
        d = tflite.DepthwiseConv2DOptions(); bo = o["op"].BuiltinOptions(); d.Init(bo.Bytes, bo.Pos); return d

    def conv_options(o):
        d = tflite.Conv2DOptions(); bo = o["op"].BuiltinOptions(); d.Init(bo.Bytes, bo.Pos); return d

    blocks = []
    i = 0
    while i < len(ops):
        e = ops[i]
        if e["type"] == "CONV_2D" and shape(e["ins"][1])[1:3] == [1, 1] and conv_options(e).StrideH() == 1:
            j = i + 1
            cur = e["outs"][0]
            hs1 = pad = hs2 = None
            def take(kind):
                nonlocal j, cur
                if j < len(ops) and ops[j]["type"] == kind and ops[j]["ins"][0] == cur and users.get(cur, 0) == 1:
                    o = ops[j]; j += 1; cur = o["outs"][0]; return o
                return None
            hs1 = take("HARD_SWISH")
            pad = take("PAD")
            d = take("DEPTHWISE_CONV_2D")
            if d is not None:
                hs2 = take("HARD_SWISH")
                mean = mulc = mulg = None
                feat = cur                       # the F2 tensor
                if j < len(ops) and ops[j]["type"] == "MEAN" and ops[j]["ins"][0] == feat and users.get(feat, 0) == 2:
                    k = j
                    mean = ops[k]; k += 1
                    ca = ops[k] if ops[k]["type"] == "CONV_2D" and ops[k]["ins"][0] == mean["outs"][0] else None; k += 1
                    cb = ops[k] if ca and ops[k]["type"] == "CONV_2D" and ops[k]["ins"][0] == ca["outs"][0] else None; k += 1
                    mc = ops[k] if cb and ops[k]["type"] == "MUL" and cb["outs"][0] in ops[k]["ins"] else None; k += 1
                    mg = ops[k] if mc and ops[k]["type"] == "MUL" and feat in ops[k]["ins"] and mc["outs"][0] in ops[k]["ins"] else None; k += 1
                    if mg is not None and all(users.get(t["outs"][0], 0) == 1 for t in (mean, ca, cb, mc, mg)) \
                            and shape(mg["ins"][0 if mg["ins"][1] == mc["outs"][0] else 1]) == shape(feat) \
                            and shape(mc["outs"][0])[1:3] == [1, 1]:
                        mulc, mulg = mc, mg
                        # MUL must not carry a fused activation (the hardware clamps with the MUL's own range anyway)
                        j = k; cur = mg["outs"][0]
                    else:
                        mean = None
                p = take("CONV_2D") if (mean is not None) or (j < len(ops) and users.get(cur, 0) == 1) else None
                if p is not None and shape(p["ins"][1])[1:3] == [1, 1] and shape(d["ins"][1])[1] in (3, 5) \
                        and shape(d["ins"][1])[1] == shape(d["ins"][1])[2] and conv_options(p).StrideH() == 1:
                    dopt = dw_options(d)
                    assert dopt.DepthMultiplier() == 1 and dopt.StrideH() == dopt.StrideW()
                    K = shape(d["ins"][1])[1]; stride = dopt.StrideH()
                    ein, eout, dout, pout = shape(e["ins"][0]), shape(e["outs"][0]), shape(d["outs"][0]), shape(p["outs"][0])
                    pt = pl = 0
                    if pad is not None:
                        pt_ = g.Tensors(pad["ins"][1]); raw = m.Buffers(pt_.Buffer()).DataAsNumpy()
                        pads = np.frombuffer(raw.tobytes(), dtype=np.int32).reshape(4, 2).tolist()
                        assert pads[0] == [0, 0] and pads[3] == [0, 0] and min(map(min, pads)) >= 0, "unsupported PAD %s" % pads
                        pt, pb, pl, pr = pads[1][0], pads[1][1], pads[2][0], pads[2][1]
                        assert dopt.Padding() == tflite.Padding.VALID, "depthwise after PAD must be VALID"
                        assert zp(pad["outs"][0]) == zp(pad["ins"][0]), "PAD must keep the zero point"
                        assert (ein[1] + pt + pb - K) // stride + 1 == dout[1] and (ein[2] + pl + pr - K) // stride + 1 == dout[2]
                    blk = dict(e=e["k"], hs1=hs1["k"] if hs1 else -1, pad=pad["k"] if pad else -1, d=d["k"],
                                       hs2=hs2["k"] if hs2 else -1, mean=mean["k"] if mean else -1,
                                       mulg=mulg["k"] if mulg else -1, p=p["k"], H=ein[1], W=ein[2], N=ein[3], M=eout[3],
                                       P=pout[3], K=K, stride=stride, oh=pout[1], ow=pout[2], pt=pt, pl=pl)
                    ok, why = fits(blk)
                    if ok: blocks.append(blk)
                    else: print("  (block at op %d NOT fused: %s)" % (blk["e"], why))
                    i = j; continue
        i += 1

    mf1 = max([b["H"] * b["W"] * b["M"] for b in blocks] + [1])
    mf2 = max([b["oh"] * b["ow"] * b["M"] for b in blocks] + [1])
    mo = max([b["oh"] * b["ow"] * b["P"] for b in blocks] + [1])
    mi = max([b["H"] * b["W"] * b["N"] for b in blocks] + [1])
    with open(sys.argv[2], "w") as f:
        f.write("// AUTO-GENERATED by tools/gen_fused_plan.py from %s -- do not edit\n" % sys.argv[1].split("/")[-1])
        f.write("#pragma once\n\n// op indices count only CONV_2D / DEPTHWISE_CONV_2D / PAD / HARD_SWISH in execution order (-1 = absent)\n")
        f.write("typedef struct { int e, hs1, pad, d, hs2, mean, mulg, p;  int H, W, N, M, P, K, stride, oh, ow, pad_top, pad_left; } fused_block_t;\n\n")
        f.write("static const fused_block_t kFusedBlocks[] = {\n")
        for b in blocks:
            f.write("  { %3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,   %3d,%3d,%4d,%4d,%4d, %d, %d, %3d,%3d, %d,%d },\n" %
                    (b["e"], b["hs1"], b["pad"], b["d"], b["hs2"], b["mean"], b["mulg"], b["p"], b["H"], b["W"], b["N"], b["M"], b["P"],
                     b["K"], b["stride"], b["oh"], b["ow"], b["pt"], b["pl"]))
        if not blocks:
            f.write("  { -1,-1,-1,-1,-1,-1,-1,-1, 0,0,0,0,0, 3,1, 0,0, 0,0 },\n")
        f.write("};\nstatic const int kNumFusedBlocks = %d;\nstatic const int kNumTrackedOps = %d;\n" % (len(blocks), len(tracked)))
        f.write("#define FUSED_MAX_F1  %d   /* largest H*W*M   (verify buffers) */\n#define FUSED_MAX_F2  %d   /* largest oh*ow*M */\n"
                "#define FUSED_MAX_OUT %d   /* largest oh*ow*P (scratch for overlapping output) */\n"
                "#define FUSED_MAX_IN  %d   /* largest H*W*N   (verify copy of the expansion input) */\n" % (mf1, mf2, mo, mi))
    if len(sys.argv) > 3:
        with open(sys.argv[3], "w") as f:
            for b in blocks:
                f.write(" ".join(str(b[k]) for k in ("e", "hs1", "pad", "d", "hs2", "mean", "mulg", "p", "H", "W", "N", "M", "P", "K", "stride", "oh", "ow", "pt", "pl")) + "\n")
    print("model ops=%d tracked=%d  fusable blocks=%d  (5x5: %d, 3x3: %d, with SE: %d)" %
          (len(ops), len(tracked), len(blocks), sum(b["K"] == 5 for b in blocks), sum(b["K"] == 3 for b in blocks), sum(b["mean"] >= 0 for b in blocks)))
    for n, b in enumerate(blocks):
        print("  block %2d  E=%3d%s%s D=%3d%s%s P=%3d   %3dx%3dx%-3d -> M=%3d -> %3dx%3dx%-3d  K%d s%d" %
              (n, b["e"], " HS=%d" % b["hs1"] if b["hs1"] >= 0 else "", " PAD=%d" % b["pad"] if b["pad"] >= 0 else "",
               b["d"], " HS=%d" % b["hs2"] if b["hs2"] >= 0 else "", " SE(MEAN=%d,MUL=%d)" % (b["mean"], b["mulg"]) if b["mean"] >= 0 else "", b["p"], b["H"], b["W"], b["N"], b["M"], b["oh"], b["ow"], b["P"], b["K"], b["stride"]))


main()
