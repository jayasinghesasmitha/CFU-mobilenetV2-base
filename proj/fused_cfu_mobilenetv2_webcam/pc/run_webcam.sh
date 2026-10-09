#!/bin/sh
# Opens the camera, then classifies one picture per minute on the board.  Pass your serial port, e.g.:
#   ./run_webcam.sh --port /dev/ttyUSB2
cd "$(dirname "$0")"
exec python3 webcam_classifier.py "$@"
