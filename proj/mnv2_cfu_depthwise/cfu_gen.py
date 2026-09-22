#!/usr/bin/env python3
"""Generate cfu.v for the fused depthwise project."""

import os.path
from amaranth.back import verilog

from gateware.fused_depthwise import make_cfu

VERILOG_FILENAME = "cfu.v"


def main():
    cfu = make_cfu()
    generated = verilog.convert(cfu, name="Cfu", ports=cfu.ports)
    previous = None
    if os.path.exists(VERILOG_FILENAME):
        with open(VERILOG_FILENAME, "r") as source:
            previous = source.read()
    if generated != previous:
        with open(VERILOG_FILENAME, "w") as output:
            output.write(generated)


if __name__ == "__main__":
    main()
