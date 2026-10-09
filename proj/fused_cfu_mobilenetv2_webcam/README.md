# fused_cfu_mobilenetv2_webcam

A webcam takes a picture **once a minute**, the picture is converted to the model's raw input format automatically,
and **MobileNetV2 a0.35 INT8 runs on the FPGA with the fused Expansion->Depthwise->Projection CFU**
(same model, same CFU and same TFLM integration as `fused_cfu_with_mobilenetv2`). The class name and cycle counts
are printed on the PC after every picture.

## Why there is a PC program

The Nexys board has no camera, so the webcam is on the PC. The PC converts each frame, sends it to the board over the
USB-UART, the board classifies it and sends the result back. The bitstream and firmware are loaded **once**; nothing is
rebuilt per picture (the earlier projects embedded the image in the firmware; this one receives it at run time).

```
 webcam --> pc/webcam_classifier.py --(224x224 RGB, UART 115200)--> VexRiscv firmware --> fused CFU (Ex-Dw-Pr)
   ^            |  saves <stamp>.jpg / .dat / .h / results.csv                                   |
   |            +<------------------- top-1, top-5, class names, cycle counts <----------------+
 opens when the program starts, one picture every 60 s
```

Sending 150,528 bytes at 115,200 baud takes about 13 s; the inference itself takes a few seconds on the board.
Both fit easily inside one minute.

## The "dat" file

Every picture is stored in `captures/` (or `--out-dir`):

| file | content |
|---|---|
| `<stamp>.jpg` | the camera frame |
| `<stamp>.dat` | raw 224 x 224 x 3 RGB bytes (150,528 bytes) - the same format as `cat_image.dat` |
| `<stamp>.h` | C header, array named `cat_image` - drop-in replacement for `src/cat_image.h` of `mnv2_baseline` / `mnv2_cfu_package` / `fused_cfu_with_mobilenetv2` |
| `<stamp>.json` | the board's answer for this picture (top-5, all 1000 outputs, counters) |
| `results.csv` | one line per picture |

The conversion is exactly the one in `tools/image_to_header.py` of the other projects (RGB, plain bilinear resize
to 224x224); converting `inputs/cat.jpg` reproduces the existing `cat_image.dat` byte for byte (checked).
So any captured picture can be fed to the CPU-only or the 1x1-CFU project by copying its `.h` over their `cat_image.h` -
that gives you the same-image comparison for live pictures too.

## Setup (once)

PC (Linux; Windows/macOS work for the Python part, the FPGA toolchain is Linux):

```
pip install -r pc/requirements.txt          # numpy pillow pyserial opencv-python
sudo usermod -aG dialout $USER              # serial port permission (log out/in once)
```

Place this folder as `CFU-mobilenetV2-base/proj/fused_cfu_mobilenetv2_webcam/`, then build and load exactly like the other
projects (`MNv2_BASELINE`, same Makefile structure):

```
conda activate cfu-common
cd proj/fused_cfu_mobilenetv2_webcam
make clean
make prog TARGET=digilent_nexys4ddr USE_VIVADO=1 UART_SPEED=115200      # bitstream, long (Vivado)
make load TARGET=digilent_nexys4ddr USE_VIVADO=1 UART_SPEED=115200      # uploads the firmware, opens litex_term
```

At the CFU-Playground menu press `3` (project menu). The firmware loads the model and prints
`@@READY mode=fused_dsc_cfu`. **Then close litex_term (Ctrl-C)** so the serial port is free - the board keeps running.
(If you forget the `3`, the PC program presses it for you.)

## Run

```
cd pc
python3 webcam_classifier.py --port /dev/ttyUSB2        # the same port you used for litex_term
```

It opens the camera immediately, takes the first picture after 3 s, then one every 60 s, and prints for each:

```
[10:15:42] picture #1  ->  captures/20261007_101542.{jpg,dat,h}   (RGB FNV1a 0x....)
   sent in 13.2 s, inference+reply 6.1 s
   >>> TOP-1: 64  green_mamba   (score 0.406250)
       #1    64  green_mamba       0.406250 ...
   whole-model cycles : 4xx,xxx,xxx   (= xxxx.x ms at 75 MHz)
   fused CFU          : 16/16 blocks, CFU busy 1,837,367 cycles, fused-path CPU xx,xxx,xxx cycles
```

Useful options: `--interval 60`, `--camera 1` (or a video file), `--count N`, `--center-crop`, `--no-preview`
(headless), `--out-dir`, `--verbose` (echo everything the board prints), `--no-send` (capture + convert only, no board).
The preview window (press q to quit) shows the live camera; it is skipped automatically if no display is available.

## Test without the board (PC simulation)

The same program can talk to a PC simulation that runs the real TFLM interpreter with the real fused-CFU RTL
(Verilator) behind the identical UART protocol:

```
scripts/run_host_test.sh                                   # builds TFLM + RTL for the PC (needs the repo layout, ~10 min first time)
cd pc && python3 webcam_classifier.py --sim --image ../inputs/cat.jpg --count 1 --no-preview
```

## Protocol (firmware side: `src/image_server.h`)

```
PC -> board : "IMG!"  uint32 length(=150528, LE)  <150528 bytes RGB>  uint32 FNV-1a (LE)      "IMG?" = ping
board -> PC : "@@READY ..." | "@@ACK ..." | "@@NAK reason=..." | console text | "@@FUSED ..." |
              "@@RESULT frame=N top1=I score_e6=S" | "@@TOP5 i:s ..." | "@@LOGITS_E6 v0,...,v999" | "@@END"
```
A corrupted transfer is detected by the checksum and re-sent (up to 3 times). Scores are the FLOAT32 outputs x 1e6
(same convention as the other projects' printouts).

## What is and is not verified

Verified on a PC: image conversion (byte-identical to the existing `cat_image.dat`), the camera loop (a video file as the
camera), the protocol and the PC program end to end against the real TFLM + RTL simulation (see the test log in the
chat / `scripts/run_host_test.sh`), and that the simulation's answer for the cat picture is the reference answer
(top-1 class 64 = green_mamba, score 0.406250). Note: `inputs/cat.jpg` is in fact a photo of a green snake, so green_mamba is the right answer.

NOT verified: a real webcam (none was available), the RISC-V firmware build, the real UART link and the real board.
The most likely first-run problems: serial port name/permissions, another program holding the camera, or litex_term still
holding the serial port.

## Notes

* The classification of a live picture is only as good as the 224x224 stretch (as in the other projects, the picture is
  resized without cropping unless you pass `--center-crop`).
* Renode is not used here (it has no webcam/UART bridge for this flow); use the board or the PC simulation.
* The cycle counts printed come from the board's own cycle counter, so they are real hardware numbers.
