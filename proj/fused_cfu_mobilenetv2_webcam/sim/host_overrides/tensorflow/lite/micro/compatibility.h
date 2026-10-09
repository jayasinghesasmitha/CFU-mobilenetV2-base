// HOST-BUILD ONLY override: modern GCC rejects TFLM's private class-specific operator delete when used with
// placement new. The RISC-V firmware build is unaffected (it uses the original header).
#ifndef TENSORFLOW_LITE_MICRO_COMPATIBILITY_H_
#define TENSORFLOW_LITE_MICRO_COMPATIBILITY_H_
#define TF_LITE_REMOVE_VIRTUAL_DELETE
#endif
