# Where the custom function unit is, and how one job flows through it

## 1. The custom instructions (rtl/cfu.v, module `Cfu`, bottom of file)
CPU side (sw/dsc_driver.c):   cfu_op0(funct7, rs1, rs2)   -> RISC-V `custom0` instruction
CFU side:                     funct7 = cmd_payload_function_id[9:3]

| funct7 | name   | rs1                           | rs2        | returns            |
|--------|--------|-------------------------------|------------|--------------------|
| 0      | PING   | -                             | -          | 0xD5C00001         |
| 1      | CFG    | config register index         | value      | -                  |
| 2      | MEMW   | {mem_id[4:0], addr[26:0]}     | 32-bit data| -                  |
| 3      | START  | -                             | -          | -                  |
| 4      | STATUS | -                             | -          | 1 if busy          |
| 5      | OUTRD  | output word index             | -          | 4 packed int8      |
| 6      | WAIT   | -                             | -          | (blocks till idle) |
| 7      | CYCLES | -                             | -          | hw busy cycles     |

Decode (cfu.v, `case (f7)`): CFG writes cfg[idx]; MEMW pulses mw_en with id/addr/data into the core;
START pulses start_p; OUTRD reads the output RAM (2-cycle wait); WAIT holds rsp_valid low while busy.

## 2. Memory map used by MEMW (mem_id)
0 expansion weights | 1,2,3 expansion bias/mult/shift | 4..12 depthwise weight banks (tap 0..8)
13,14,15 depthwise bias/mult/shift | 16 projection weights (addr = engine<<8 | word)
17,18,19 projection bias/mult/shift | 20 input map (addr = bank<<23 | word); bank = (row%3)*3 + (col%3)

## 3. One job, step by step (software = dsc_driver.c, hardware = cfu.v)
1. dsc_load_layer(): 24 CFG writes, then MEMW for every weight / bias / multiplier / shift.
2. dsc_load_ifmap(): for each pixel compute its bank and slot, MEMW its N/4 words.
3. dsc_start(): START. Core FSM: IDLE -> INIT1 -> INIT2 -> (PREP0 -> PREP1 -> ISSUE [-> GAP]) per pixel -> DRAIN.
4. PREP0/PREP1: window rows/cols r0..r0+2, c0..c0+2, bounds -> valid mask; bank addresses via div-by-3.
5. ISSUE: every clock one (channel m, 8-wide chunk) enters the pipeline together with a "tag"
   (m, first/last flags, valid mask, row/col mod 3) that travels beside the data.
6. Pipeline (X = clocks after issue):
   X1  read 9 input banks + expansion weights      X10 F1 (9 int8 values) ready
   X2  72 multiplies (9 engines x 8 MACs)           X11 9 depthwise multiplies (padding taps masked to 0)
   X3  sum 8 products per engine                    X12 sum 9 products -> depthwise requantize
   X4  accumulate over chunks -> full sum           X18 F2 (1 int8 value) ready
   X4..X10 bias + requantize (6 stages)             X19 multiply F2 by each projection weight
                                                    X20 accumulate over the M channels
7. After the last channel, the P sums are copied to a shift register and requantized one per clock,
   packed 4 bytes per word into the output RAM (NHWC order).
8. dsc_wait(): WAIT returns when the last word is written. dsc_read_output(): OUTRD per word.

## 4. Why no F1/F2 buffer exists
F1 and F2 only live in pipeline registers (`f1q_flat`, `dq`, `pprod`, `pacc`). Per output pixel the CFU
recomputes the expansion of its 3x3 window (up to 9x redundant work, done 72 MACs/clock) instead of
storing a 40x40x48 map.

## 5. Cycle count
hw_busy = 31 + P + pixels*(2 + M*N/8) + (pixels-1)*max(0, P+4 - M*N/8)
(read it from hardware with CYCLES; the menu prints it next to this prediction)
