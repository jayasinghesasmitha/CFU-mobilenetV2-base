#!/usr/bin/env python3
"""
compare_runs.py -- compare the console logs of two MobileNetV2 runs on the same image
(e.g.  mnv2_cfu_package = existing 1x1 CFU   vs   fused_cfu_with_mobilenetv2 = fused DSC CFU).

usage:  python3 tools/compare_runs.py existing_cfu.log fused_cfu.log [cpu_only.log]

Save each run's UART output to a text file first (Renode window log, litex_term output, ...).
Prints: top-1 class, output statistics, the first 20 FLOAT32 outputs (must be identical), total cycles
and the speedup between the runs.
"""
import re, sys

def parse(path):
    t = open(path, errors="replace").read()
    t = re.sub(r"\x1b\[[0-9;]*m", "", t)
    # Renode prefixes UART lines with a timestamp/tag; strip it so the patterns below work on both
    t = re.sub(r"^.*?\[INFO\] uart: \[[^\]]*\]\s?", "", t, flags=re.M)
    r = {}
    m = re.search(r"Top-1 class index\s*:\s*(-?\d+)", t);              r["top1"] = int(m.group(1)) if m else None
    m = re.search(r"Top-1 score\s*:\s*(-?\d+)", t);                    r["score"] = int(m.group(1)) if m else None
    for k in ("Min", "Max", "Mean"):
        m = re.search(r"%s\s*=\s*(-?\d+) x 10\^-6" % k, t);            r[k.lower()] = int(m.group(1)) if m else None
    r["out20"] = [int(x) for x in re.findall(r"output\[\d+\]\s*=\s*(-?\d+) x 10\^-6", t)][:20]
    m = re.findall(r"\(\s*(\d+)\s*\)\s*cycles total", t) or re.findall(r"(\d+)\s+cycles total", t)
    r["cycles"] = int(m[-1]) if m else None
    m = re.search(r"CFU busy cycles\s*:\s*(\d+)", t);                  r["cfu_busy"] = int(m.group(1)) if m else None
    m = re.search(r"Blocks run on CFU\s*:\s*(\d+) / (\d+)", t);        r["blocks"] = m.groups() if m else None
    m = re.search(r"Verification vs C reference: (\d+) blocks checked, (\d+) mismatching bytes -> (\w+)", t)
    r["verify"] = m.groups() if m else None
    m = re.search(r"RGB FNV1a\s*:\s*(0x[0-9a-f]+)", t);                r["img"] = m.group(1) if m else None
    m = re.search(r"Mode\s*:\s*(.*)", t);                              r["mode"] = m.group(1).strip() if m else "?"
    return r

def main():
    if len(sys.argv) < 3: print(__doc__); return 1
    paths = sys.argv[1:]
    labels = ["existing 1x1 CFU", "fused DSC CFU", "CPU only"][:len(paths)]
    runs = [parse(p) for p in paths]
    w = 30
    print("%-26s" % "" + "".join("%-*s" % (w, l) for l in labels))
    def row(name, key, fmt=str):
        print("%-26s" % name + "".join("%-*s" % (w, fmt(r[key]) if r[key] is not None else "-") for r in runs))
    row("mode (from log)", "mode"); row("image FNV1a", "img"); row("top-1 class", "top1"); row("top-1 score (x1e-6)", "score")
    row("output min (x1e-6)", "min"); row("output max (x1e-6)", "max"); row("output mean (x1e-6)", "mean")
    row("blocks on fused CFU", "blocks", lambda v: "%s / %s" % v)
    row("CFU busy cycles (fused)", "cfu_busy"); row("total cycles (whole model)", "cycles", lambda v: "{:,}".format(v))
    if all(r["verify"] for r in runs[1:2]): row("fused verify vs C ref", "verify", lambda v: "%s blk, %s bad, %s" % v)
    same = all(runs[0]["out20"] == r["out20"] and r["out20"] for r in runs[1:])
    print("\nfirst-20 FLOAT32 outputs identical across runs : %s" % ("YES" if same else "NO  <-- investigate"))
    same_t = all(runs[0]["top1"] == r["top1"] for r in runs[1:])
    print("top-1 class identical across runs             : %s" % ("YES" if same_t else "NO  <-- investigate"))
    cyc = [r["cycles"] for r in runs]
    if all(cyc):
        print("\ncycle comparison (lower is better):")
        for l, c in zip(labels, cyc): print("   %-18s %14s cycles" % (l, "{:,}".format(c)))
        base = cyc[-1] if len(runs) == 3 else cyc[0]
        bl = labels[-1] if len(runs) == 3 else labels[0]
        for l, c in zip(labels, cyc):
            if c != base: print("   %-18s speedup vs %-18s : %.2fx" % (l, bl, base / c))
        if len(runs) >= 2: print("   fused DSC CFU vs existing 1x1 CFU          : %.2fx (%s)" % (cyc[0] / cyc[1], "fused faster" if cyc[1] < cyc[0] else "existing faster"))
    return 0

sys.exit(main())
