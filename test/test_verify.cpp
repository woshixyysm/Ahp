//===- test_verify.cpp - Compile-only pass verification test --------------===//
//
// This file is compiled by the CMake LLVM tests (hetero-opt-elite /
// hetero-opt-ampere) as a smoke test for the pass plugin.  It contains
// representative kernels whose IR structure exercises each code path in
// AArch64HeteroOptPass:
//
//   1. vec_add      – sequential streaming, small stride → Streaming
//   2. scale_array  – strided 1D, single outer loop → ModerateReuse
//   3. matmul_tile  – inner loop with outer-loop reuse → HighReuse
//   4. list_sum     – indirect pointer chase (linked list) → Indirect
//   5. memset_large – unit-stride store loop → Streaming
//   6. invariant_load_test – loop-invariant load + streaming array
//
// Each kernel is annotated with its expected prefetch classification.
// The pass is verified by compiling with -fpass-plugin and checking
// that compilation succeeds (no crashes, no miscompiles).
//
//===----------------------------------------------------------------------===//

#include <cstddef>
#include <cstdint>

// ── 1. Sequential streaming ──────────────────────────────────────────────────
// Expected: Streaming (|stride|=4, no outer loop)
//   Oryon  → PLDL1STRM (locality=1)
//   Ampere → skip (HW PF handles it)
void vec_add(float *__restrict__ dst,
             const float *__restrict__ src1,
             const float *__restrict__ src2,
             size_t n) {
  for (size_t i = 0; i < n; i++)
    dst[i] = src1[i] + src2[i];
}

// ── 2. Strided scale ─────────────────────────────────────────────────────────
// Expected: ModerateReuse (|stride|=4, no outer loop, not purely streaming)
void scale_array(int32_t *__restrict__ dst, const int32_t *__restrict__ src,
                 size_t n, int factor) {
  for (size_t i = 0; i < n; i++)
    dst[i] = src[i] * factor;
}

// ── 3. Matrix multiply (tiled, compile-time trip counts) ─────────────────────
//
// Design decisions:
// (a) Compile-time tile size → constant backedge TC → no SVE guard trigger.
// (b) acc[TILE] fits in NEON registers; provenance check (isProvablyHeapOrGlobal)
//     will skip acc[] (local alloca) but will instrument A[] and B[] args.
// (c) Expected classifications:
//     A[(ii+i)*N + kk+k] in inner k-loop: outer-loop reuse → HighReuse
//     B[(kk+k)*N + jj+j] in inner j-loop: one-pass per k → Streaming
//
static constexpr size_t TILE = 4;

__attribute__((noinline))
static void matmul_tile(float *__restrict__ C,
                        const float *__restrict__ A,
                        const float *__restrict__ B,
                        size_t ii, size_t jj, size_t kk, size_t N) {
  float acc[TILE] = {};
  for (size_t k = 0; k < TILE; k++) {
    float a = A[(ii * N) + kk + k];
    for (size_t j = 0; j < TILE; j++)
      acc[j] += a * B[(kk + k) * N + jj + j];
  }
  for (size_t j = 0; j < TILE; j++)
    C[ii * N + jj + j] += acc[j];
}

void mat_mul_kernel(float *__restrict__ C,
                    const float *__restrict__ A,
                    const float *__restrict__ B,
                    size_t N) {
  const size_t N_t = (N / TILE) * TILE;
  for (size_t ii = 0; ii < N_t; ii++)
    for (size_t jj = 0; jj < N_t; jj += TILE)
      for (size_t kk = 0; kk < N_t; kk += TILE)
        matmul_tile(C, A, B, ii, jj, kk, N);

  // Scalar remainder: N % TILE != 0.
  for (size_t i = 0; i < N; i++) {
    for (size_t j = N_t; j < N; j++) {
      float sum = 0.f;
      for (size_t k = 0; k < N; k++)
        sum += A[i * N + k] * B[k * N + j];
      C[i * N + j] += sum;
    }
  }
}

// ── 4. Linked-list traversal – indirect pointer chase ────────────────────────
// Expected: AccessPattern::Indirect in AArch64PrefetchAnalysis.
// The function-level pass (AArch64HeteroOptPass) skips non-affine SCEVs;
// the loop-level pass (AArch64PrefetchInjector) handles this pattern.
struct Node { Node *next; int value; };

int list_sum(Node *head) {
  int sum = 0;
  while (head) {
    sum += head->value;
    head = head->next;
  }
  return sum;
}

// ── 5. Streaming store ────────────────────────────────────────────────────────
// Expected: Streaming (|stride|=4, no outer loop)
void memset_large(float *__restrict__ dst, float val, size_t n) {
  for (size_t i = 0; i < n; i++)
    dst[i] = val;
}

// ── 6. Loop-invariant load + streaming array ──────────────────────────────────
// Expected:
//   arr[i] → Streaming (sequential, |stride|=4)
//   *table → Invariant (loop-invariant load, skipped)
int invariant_load_test(const int *__restrict__ arr, size_t n,
                        const int *__restrict__ table) {
  int sum = 0;
  for (size_t i = 0; i < n; i++)
    sum += arr[i] + *table;
  return sum;
}
