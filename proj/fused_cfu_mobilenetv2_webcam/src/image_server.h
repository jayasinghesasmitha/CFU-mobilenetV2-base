// image_server.h -- UART image link shared by the board firmware and the PC test harness.
//
// The PC program sends one 224x224x3 RGB image (raw bytes, row-major, R G B) per request:
//
//   PC -> board :  'I' 'M' 'G' '!'   uint32 length (LE, must be 150528)   <length bytes>   uint32 FNV-1a (LE)
//   PC -> board :  'I' 'M' 'G' '?'   (ping)  ->  board answers "@@READY ..."
//   board -> PC :  "@@ACK ..." (image accepted) | "@@NAK reason=..." (resend)
//                  ... free-form console text (TFLM profile etc.) ...
//                  "@@FUSED blocks=.. hw_cycles=.. cpu_cycles=.."      (optional)
//                  "@@RESULT frame=N top1=I score_e6=S"
//                  "@@TOP5 i:score i:score ..."
//                  "@@LOGITS_E6 v0,v1,...,v999"
//                  "@@END"
// Scores are the FLOAT32 model outputs multiplied by 1e6 and truncated (same convention as the other projects).
// Only getchar()/printf()/putchar() are used, so the same code runs on the VexRiscv console and on a PC pipe.
#pragma once
#include <stdint.h>
#include <stdio.h>

namespace imgsrv {

constexpr int kW = 224, kH = 224, kC = 3;
constexpr uint32_t kBytes = kW * kH * kC;
constexpr int kClasses = 1000;

// Runs the network on one RGB image. Returns the FLOAT32 output tensor (kClasses values) or nullptr on failure.
typedef const float* (*InferFn)(const uint8_t* rgb);

static inline uint32_t Fnv1a(const uint8_t* d, uint32_t n) {
  uint32_t h = 2166136261u;
  for (uint32_t i = 0; i < n; i++) { h ^= d[i]; h *= 16777619u; }
  return h;
}

static inline bool ReadByte(uint8_t* b) {
  int c = getchar();
  if (c < 0) return false;
  *b = (uint8_t)c;
  return true;
}

static inline bool ReadU32(uint32_t* v) {
  uint32_t x = 0;
  for (int i = 0; i < 4; i++) {
    uint8_t b;
    if (!ReadByte(&b)) return false;
    x |= (uint32_t)b << (8 * i);
  }
  *v = x;
  return true;
}

static inline void PrintResult(uint32_t frame, const float* out) {
  int top[5] = {-1, -1, -1, -1, -1};
  for (int k = 0; k < 5; k++) {                      // 5 selection passes (n = 1000, trivial cost)
    int best = -1;
    for (int i = 0; i < kClasses; i++) {
      bool used = false;
      for (int j = 0; j < k; j++) if (top[j] == i) used = true;
      if (!used && (best < 0 || out[i] > out[best])) best = i;
    }
    top[k] = best;
  }
  printf("@@RESULT frame=%lu top1=%d score_e6=%ld\n", (unsigned long)frame, top[0],
         (long)(int32_t)(out[top[0]] * 1000000.0f));
  printf("@@TOP5");
  for (int k = 0; k < 5; k++) printf(" %d:%ld", top[k], (long)(int32_t)(out[top[k]] * 1000000.0f));
  printf("\n@@LOGITS_E6 ");
  for (int i = 0; i < kClasses; i++) printf(i ? ",%ld" : "%ld", (long)(int32_t)(out[i] * 1000000.0f));
  printf("\n@@END\n");
  fflush(stdout);
}

// Never returns on the board; returns when the input stream ends (PC test harness).
static inline void Serve(InferFn infer, const char* mode) {
  static uint8_t frame[kBytes];
  uint32_t count = 0;
  printf("\n@@READY mode=%s bytes=%lu\n", mode, (unsigned long)kBytes);
  fflush(stdout);
  for (;;) {
    uint32_t win = 0;
    for (;;) {                                        // resynchronise on the 4-byte magic
      uint8_t c;
      if (!ReadByte(&c)) return;
      win = (win << 8) | c;
      if (win == 0x494D4721u) break;                  // "IMG!"
      if (win == 0x494D473Fu) {                       // "IMG?"
        printf("@@READY mode=%s bytes=%lu\n", mode, (unsigned long)kBytes);
        fflush(stdout);
        win = 0;
      }
    }
    uint32_t len;
    if (!ReadU32(&len)) return;
    if (len != kBytes) { printf("@@NAK reason=length got=%lu\n", (unsigned long)len); fflush(stdout); continue; }
    for (uint32_t i = 0; i < len; i++) if (!ReadByte(&frame[i])) return;
    uint32_t sent;
    if (!ReadU32(&sent)) return;
    const uint32_t calc = Fnv1a(frame, len);
    if (sent != calc) {
      printf("@@NAK reason=checksum sent=0x%08lx calc=0x%08lx\n", (unsigned long)sent, (unsigned long)calc);
      fflush(stdout);
      continue;
    }
    printf("@@ACK len=%lu fnv=0x%08lx\n", (unsigned long)len, (unsigned long)calc);
    fflush(stdout);
    const float* out = infer(frame);
    if (!out) { printf("@@NAK reason=inference_failed\n@@END\n"); fflush(stdout); continue; }
    PrintResult(++count, out);
  }
}

}  // namespace imgsrv
