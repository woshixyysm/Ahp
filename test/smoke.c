/* smoke.c - Simple test input for AArch64HeteroOpt */
/* Compile with: clang -mcpu=oryon-1 -O2 -fpass-plugin=./AArch64HeteroOpt.so */

#include <stdint.h>
#include <stddef.h>

void vec_add(float* __restrict dst,
             const float* __restrict src1,
             const float* __restrict src2,
             size_t n) {
  for (size_t i = 0; i < n; i++) {
    dst[i] = src1[i] + src2[i];
  }
}

void mat_mul_blocked(float* __restrict C,
                     const float* __restrict A,
                     const float* __restrict B,
                     size_t N, size_t block) {
  for (size_t ii = 0; ii < N; ii += block) {
    for (size_t jj = 0; jj < N; jj += block) {
      for (size_t kk = 0; kk < N; kk += block) {
        for (size_t i = ii; i < ii + block && i < N; i++) {
          for (size_t j = jj; j < jj + block && j < N; j++) {
            float sum = C[i * N + j];
            for (size_t k = kk; k < kk + block && k < N; k++) {
              sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
          }
        }
      }
    }
  }
}

struct Node {
  struct Node* next;
  int value;
};

int list_sum(struct Node* head) {
  int sum = 0;
  while (head) {
    sum += head->value;
    head = head->next;
  }
  return sum;
}

int main() {
  float a[256], b[256], c[256];
  for (int i = 0; i < 256; i++) {
    a[i] = (float)i;
    b[i] = (float)(256 - i);
  }
  vec_add(c, a, b, 256);
  return 0;
}
