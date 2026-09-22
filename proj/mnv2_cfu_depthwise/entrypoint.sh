#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
PYRUN="${ROOT}/scripts/pyrun"
cd "${HERE}"

run_python_tests() {
  "${PYRUN}" -m unittest discover -s gateware -p 'test_*.py'
  "${PYRUN}" -m unittest discover -s tests -p 'test_*.py'
}

case "${1:-test}" in
  test)
    # Python/gateware verification is always available from the checked-in
    # repository environment. Host C++ checks are added when a compiler exists.
    run_python_tests
    if command -v g++ >/dev/null 2>&1; then
      mkdir -p build/tests
      g++ -std=c++11 -Wall -Wextra -Werror -DCFU_SOFTWARE_DEFINED \
        -I"${ROOT}/common/src" -Isrc \
        src/fused_depthwise.cc src/software_cfu.cc tests/test_cxx_reference.cc \
        -o build/tests/test_cxx_reference
      build/tests/test_cxx_reference
      g++ -std=c++11 -Wall -Wextra -Werror -DCFU_SOFTWARE_DEFINED \
        -I"${ROOT}/common/src" -Isrc \
        src/fused_depthwise.cc src/software_cfu.cc tests/test_software_cfu.cc \
        -o build/tests/test_software_cfu
      build/tests/test_software_cfu
    else
      echo "NOTE: g++ is unavailable; C++ host tests were not run." >&2
    fi
    ;;
  generate)
    "${PYRUN}" cfu_gen.py
    ;;
  image)
    if [[ $# -ne 2 ]]; then
      echo "usage: $0 image /path/to/image.jpg" >&2
      exit 2
    fi
    "${PYRUN}" tools/image_to_header.py "$2" src/input_image.h
    ;;
  build)
    if ! command -v make >/dev/null 2>&1; then
      echo "ERROR: GNU Make is required for the CFU-Playground firmware build." >&2
      exit 127
    fi
    make software -j"${JOBS:-$(nproc)}"
    ;;
  clean)
    if command -v make >/dev/null 2>&1; then
      make clean
    else
      rm -rf build cfu.v
    fi
    ;;
  *)
    echo "usage: $0 {test|generate|image /path/to/image.jpg|build|clean}" >&2
    exit 2
    ;;
esac
