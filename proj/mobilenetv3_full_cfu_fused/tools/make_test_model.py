#!/usr/bin/env python3
"""
make_test_model.py -- writes a small INT8 TFLite model (raw flatbuffer, no TensorFlow needed) that contains the
MobileNetV3 features the uploaded 'minimalistic' model does NOT have, so the fused-CFU extension can be tested
end to end through the real TFLM interpreter:

   input 24x24x16
   block A  E 16->96 relu | DW 3x3 s1 SAME relu           | P 96->24
   block B  E 24->96      | HARD_SWISH | DW 5x5 s1 SAME   | HARD_SWISH | P 96->24
   block C  E 24->72 relu | PAD(1,2,1,2) | DW 5x5 s2 VALID relu | P 72->32            (explicit PAD op, like Keras)
   block D  E 32->96 relu | DW 3x3 s2 SAME relu           | P 96->40
   block E  E 40->96 | HS | DW 5x5 s1 | HS | SE | P 96->24          squeeze-and-excite (MEAN, 2 convs, MUL, MUL)
   block F  E 24->96 relu | DW 3x3 s1 relu | SE (MEAN keeps the input quantisation) | P 96->72   (72 > 56: two output groups)
   block G  E 72->96 | HS | DW 5x5 s2 | HS | SE | P 96->32

usage: python3 tools/make_test_model.py sim/test_models/mixed_k35_hswish.tflite [seed]
"""
import sys
import numpy as np
import flatbuffers
import tflite

B = tflite.BuiltinOperator
BO = tflite.BuiltinOptions
INT8, INT32 = 9, 2
RELU6 = 3
NONE, RELU = 0, 1
SAME, VALID = 0, 1


def end_vec(b, n):
    try:
        return b.EndVector(n)
    except TypeError:
        return b.EndVector()


def vec_i32(b, xs):
    b.StartVector(4, len(xs), 4)
    for x in reversed(xs): b.PrependInt32(int(x))
    return end_vec(b, len(xs))


def vec_f32(b, xs):
    b.StartVector(4, len(xs), 4)
    for x in reversed(xs): b.PrependFloat32(float(x))
    return end_vec(b, len(xs))


def vec_i64(b, xs):
    b.StartVector(8, len(xs), 8)
    for x in reversed(xs): b.PrependInt64(int(x))
    return end_vec(b, len(xs))


def vec_off(b, offs):
    b.StartVector(4, len(offs), 4)
    for o in reversed(offs): b.PrependUOffsetTRelative(o)
    return end_vec(b, len(offs))


class Graph:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.buffers = [b""]          # buffer 0 = empty
        self.tensors = []             # dict(shape, type, buffer, name, scale[], zp[], qdim)
        self.ops = []                 # dict(code, ins, outs, opt_type, opt)
        self.codes = []

    def tensor(self, name, shape, typ=INT8, data=None, scale=None, zp=None, qdim=0):
        buf = 0
        if data is not None:
            self.buffers.append(data.tobytes()); buf = len(self.buffers) - 1
        self.tensors.append(dict(shape=list(shape), type=typ, buffer=buf, name=name,
                                 scale=None if scale is None else list(np.atleast_1d(scale)),
                                 zp=None if zp is None else list(np.atleast_1d(zp)), qdim=qdim))
        return len(self.tensors) - 1

    def op(self, code, ins, outs, opt_type=0, opt=None):
        if code not in self.codes: self.codes.append(code)
        self.ops.append(dict(code=code, ins=ins, outs=outs, opt_type=opt_type, opt=opt))

    # ---- layers (return output tensor index, scale, zp)
    def conv(self, x, xs, xz, O, act, so, zo, name):
        I = self.tensors[x]["shape"][3]
        w = np.clip(self.rng.normal(0, 22, (O, 1, 1, I)).round(), -127, 127).astype(np.int8)
        sf = self.rng.uniform(0.004, 0.009, O).astype(np.float32)
        wt = self.tensor(name + "_w", w.shape, INT8, w, sf, np.zeros(O, np.int64), 0)
        bias = self.rng.integers(-3000, 3000, O).astype(np.int32)
        bt = self.tensor(name + "_b", (O,), INT32, bias, (xs * sf).astype(np.float32), np.zeros(O, np.int64), 0)
        sh = self.tensors[x]["shape"]
        y = self.tensor(name, (1, sh[1], sh[2], O), INT8, None, so, zo)
        self.op(B.CONV_2D, [x, wt, bt], [y], BO.Conv2DOptions, dict(kind="conv", pad=SAME, stride=1, act=act))
        return y

    def dw(self, x, xs, xz, K, stride, pad, act, so, zo, name, out_hw):
        M = self.tensors[x]["shape"][3]
        w = np.clip(self.rng.normal(0, 22 if K == 3 else 16, (1, K, K, M)).round(), -127, 127).astype(np.int8)
        sf = self.rng.uniform(0.004, 0.009, M).astype(np.float32)
        wt = self.tensor(name + "_w", w.shape, INT8, w, sf, np.zeros(M, np.int64), 3)
        bias = self.rng.integers(-3000, 3000, M).astype(np.int32)
        bt = self.tensor(name + "_b", (M,), INT32, bias, (xs * sf).astype(np.float32), np.zeros(M, np.int64), 0)
        y = self.tensor(name, (1, out_hw[0], out_hw[1], M), INT8, None, so, zo)
        self.op(B.DEPTHWISE_CONV_2D, [x, wt, bt], [y], BO.DepthwiseConv2DOptions,
                dict(kind="dw", pad=pad, stride=stride, act=act))
        return y

    def hswish(self, x, xs, xz, so, zo, name):
        sh = self.tensors[x]["shape"]
        y = self.tensor(name, sh, INT8, None, so, zo)
        self.op(B.HARD_SWISH, [x], [y], BO.HardSwishOptions, dict(kind="hs"))
        return y

    def mean_hw(self, x, so, zo, name):
        sh = self.tensors[x]["shape"]
        ax = self.tensor(name + "_axes", (2,), INT32, np.array([1, 2], np.int32))
        y = self.tensor(name, (1, 1, 1, sh[3]), INT8, None, so, zo)
        self.op(B.MEAN, [x, ax], [y], BO.ReducerOptions, dict(kind="mean"))
        return y

    def mul(self, a, b, so, zo, name, shape):
        y = self.tensor(name, shape, INT8, None, so, zo)
        self.op(B.MUL, [a, b], [y], BO.MulOptions, dict(kind="mul"))
        return y

    def se(self, f, fs, fz, R, name, mean_same_quant=False):
        """squeeze-and-excite on feature map f (scale fs, zp fz): returns the gated map and its (scale, zp)"""
        sh = self.tensors[f]["shape"]; M = sh[3]
        ms, mz = (fs, fz) if mean_same_quant else (0.03, -117)
        m = self.mean_hw(f, ms, mz, name + "_mean")
        a = self.conv(m, ms, mz, R, RELU, 0.02, -128, name + "_fc1")
        b = self.conv(a, 0.02, -128, M, RELU6, 0.02353, -128, name + "_fc2")
        c = self.tensor(name + "_c", (), INT8, np.array([21], np.int8), 0.0078, -128)       # scalar constant
        g = self.mul(b, c, 0.003922, -128, name + "_gate", (1, 1, 1, M))
        so, zo = 0.05, -100
        y = self.mul(f, g, so, zo, name + "_mul", sh)
        return y, so, zo

    def pad(self, x, pads, name):
        t = self.tensors[x]; sh = t["shape"]
        pt = self.tensor(name + "_pads", (4, 2), INT32, np.array(pads, np.int32))
        y = self.tensor(name, (1, sh[1] + pads[1][0] + pads[1][1], sh[2] + pads[2][0] + pads[2][1], sh[3]), INT8,
                        None, t["scale"], t["zp"])
        self.op(B.PAD, [x, pt], [y], BO.PadOptions, dict(kind="pad"))
        return y

    # ---- serialisation
    def build(self, inputs, outputs):
        b = flatbuffers.Builder(1 << 20)
        buf_offs = []
        for d in self.buffers:
            data = b.CreateByteVector(d) if len(d) else None
            b.StartObject(1)
            if data is not None: b.PrependUOffsetTRelativeSlot(0, data, 0)
            buf_offs.append(b.EndObject())
        buffers = vec_off(b, buf_offs)

        t_offs = []
        for t in self.tensors:
            shape = vec_i32(b, t["shape"]); name = b.CreateString(t["name"]); q = None
            if t["scale"] is not None:
                sc = vec_f32(b, t["scale"]); zp = vec_i64(b, t["zp"])
                b.StartObject(7); b.PrependUOffsetTRelativeSlot(2, sc, 0); b.PrependUOffsetTRelativeSlot(3, zp, 0)
                b.PrependInt32Slot(6, t["qdim"], 0); q = b.EndObject()
            b.StartObject(8)
            b.PrependUOffsetTRelativeSlot(0, shape, 0); b.PrependInt8Slot(1, t["type"], 0)
            b.PrependUint32Slot(2, t["buffer"], 0); b.PrependUOffsetTRelativeSlot(3, name, 0)
            if q is not None: b.PrependUOffsetTRelativeSlot(4, q, 0)
            t_offs.append(b.EndObject())
        tensors = vec_off(b, t_offs)

        c_offs = []
        for c in self.codes:
            b.StartObject(4); b.PrependInt8Slot(0, min(c, 127), 0); b.PrependInt32Slot(2, 1, 1)
            b.PrependInt32Slot(3, c, 0); c_offs.append(b.EndObject())
        codes = vec_off(b, c_offs)

        o_offs = []
        for o in self.ops:
            ins = vec_i32(b, o["ins"]); outs = vec_i32(b, o["outs"]); opt = None
            k = o["opt"]["kind"]
            if k in ("conv", "dw"):
                b.StartObject(6 if k == "conv" else 7)
                b.PrependInt8Slot(0, o["opt"]["pad"], 0); b.PrependInt32Slot(1, o["opt"]["stride"], 0)
                b.PrependInt32Slot(2, o["opt"]["stride"], 0)
                if k == "conv":
                    b.PrependInt8Slot(3, o["opt"]["act"], 0)
                else:
                    b.PrependInt32Slot(3, 1, 0); b.PrependInt8Slot(4, o["opt"]["act"], 0)
                opt = b.EndObject()
            elif k == "mean":
                b.StartObject(1); b.PrependBoolSlot(0, True, False); opt = b.EndObject()
            elif k == "mul":
                b.StartObject(1); b.PrependInt8Slot(0, 0, 0); opt = b.EndObject()
            else:
                b.StartObject(0); opt = b.EndObject()
            b.StartObject(9)
            b.PrependUint32Slot(0, self.codes.index(o["code"]), 0)
            b.PrependUOffsetTRelativeSlot(1, ins, 0); b.PrependUOffsetTRelativeSlot(2, outs, 0)
            b.PrependUint8Slot(3, o["opt_type"], 0); b.PrependUOffsetTRelativeSlot(4, opt, 0)
            o_offs.append(b.EndObject())
        ops = vec_off(b, o_offs)

        sg_in = vec_i32(b, inputs); sg_out = vec_i32(b, outputs); sg_name = b.CreateString("main")
        b.StartObject(5)
        b.PrependUOffsetTRelativeSlot(0, tensors, 0); b.PrependUOffsetTRelativeSlot(1, sg_in, 0)
        b.PrependUOffsetTRelativeSlot(2, sg_out, 0); b.PrependUOffsetTRelativeSlot(3, ops, 0)
        b.PrependUOffsetTRelativeSlot(4, sg_name, 0)
        sg = b.EndObject()
        subgraphs = vec_off(b, [sg])
        desc = b.CreateString("fused-CFU synthetic test model")
        b.StartObject(7)
        b.PrependUint32Slot(0, 3, 0); b.PrependUOffsetTRelativeSlot(1, codes, 0)
        b.PrependUOffsetTRelativeSlot(2, subgraphs, 0); b.PrependUOffsetTRelativeSlot(3, desc, 0)
        b.PrependUOffsetTRelativeSlot(4, buffers, 0)
        model = b.EndObject()
        b.Finish(model, b"TFL3")
        return bytes(b.Output())


def main():
    out = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    g = Graph(seed)
    xs, xz = 0.05, -3
    x = g.tensor("input", (1, 24, 24, 16), INT8, None, xs, xz)
    # block A : 3x3 s1, ReLU
    e = g.conv(x, xs, xz, 96, RELU, 0.06, -128, "A_e")
    d = g.dw(e, 0.06, -128, 3, 1, SAME, RELU, 0.05, -128, "A_d", (24, 24))
    p = g.conv(d, 0.05, -128, 24, NONE, 0.08, 5, "A_p")
    # block B : 5x5 s1, hard-swish before and after the depthwise
    e = g.conv(p, 0.08, 5, 96, NONE, 0.07, -9, "B_e")
    h1 = g.hswish(e, 0.07, -9, 0.04, -120, "B_hs1")
    d = g.dw(h1, 0.04, -120, 5, 1, SAME, NONE, 0.09, 11, "B_d", (24, 24))
    h2 = g.hswish(d, 0.09, 11, 0.05, -115, "B_hs2")
    p = g.conv(h2, 0.05, -115, 24, NONE, 0.10, -6, "B_p")
    # block C : explicit PAD (1,2) then 5x5 stride 2 VALID
    e = g.conv(p, 0.10, -6, 72, RELU, 0.06, -128, "C_e")
    pd = g.pad(e, [[0, 0], [1, 2], [1, 2], [0, 0]], "C_pad")
    d = g.dw(pd, 0.06, -128, 5, 2, VALID, RELU, 0.06, -128, "C_d", (12, 12))
    p = g.conv(d, 0.06, -128, 32, NONE, 0.09, 8, "C_p")
    # block D : 3x3 stride 2 SAME
    e = g.conv(p, 0.09, 8, 96, RELU, 0.06, -128, "D_e")
    d = g.dw(e, 0.06, -128, 3, 2, SAME, RELU, 0.05, -128, "D_d", (6, 6))
    p = g.conv(d, 0.05, -128, 40, NONE, 0.12, 3, "D_p")
    # block E : 5x5 s1, hard-swish both sides, squeeze-and-excite
    e = g.conv(p, 0.12, 3, 96, NONE, 0.07, -9, "E_e")
    h1 = g.hswish(e, 0.07, -9, 0.04, -120, "E_hs1")
    d = g.dw(h1, 0.04, -120, 5, 1, SAME, NONE, 0.09, 11, "E_d", (6, 6))
    h2 = g.hswish(d, 0.09, 11, 0.05, -115, "E_hs2")
    f, fs, fz = g.se(h2, 0.05, -115, 24, "E_se")
    p = g.conv(f, fs, fz, 24, NONE, 0.10, -6, "E_p")
    # block F : 3x3 s1 relu, SE whose MEAN keeps the input quantisation (integer mean path), P = 72 (two groups)
    e = g.conv(p, 0.10, -6, 96, RELU, 0.06, -128, "F_e")
    d = g.dw(e, 0.06, -128, 3, 1, SAME, RELU, 0.05, -128, "F_d", (6, 6))
    f, fs, fz = g.se(d, 0.05, -128, 24, "F_se", mean_same_quant=True)
    p = g.conv(f, fs, fz, 72, NONE, 0.09, 8, "F_p")
    # block G : 5x5 s2 SAME, hard-swish, SE
    e = g.conv(p, 0.09, 8, 96, NONE, 0.07, -9, "G_e")
    h1 = g.hswish(e, 0.07, -9, 0.04, -120, "G_hs1")
    d = g.dw(h1, 0.04, -120, 5, 2, SAME, NONE, 0.09, 11, "G_d", (3, 3))
    h2 = g.hswish(d, 0.09, 11, 0.05, -115, "G_hs2")
    f, fs, fz = g.se(h2, 0.05, -115, 24, "G_se")
    p = g.conv(f, fs, fz, 32, NONE, 0.10, 5, "G_p")
    open(out, "wb").write(g.build([x], [p]))
    print("wrote", out, "tensors", len(g.tensors), "ops", len(g.ops))


main()
