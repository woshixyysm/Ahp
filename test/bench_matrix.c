/* Ahp benchmark: does the prefetch pass actually help on a big matrix?
 *
 * Sections:
 *   dgemm_ikj  - cache-friendly, unit-stride inner loop (the vectorizer's
 *                favourite; the pass skips vectorized loops, so this is the
 *                "control" case)
 *   dgemm_ijk  - inner loop over k: A unit stride, B stride N*8 bytes, FP
 *                reduction so it stays SCALAR without -ffast-math. This is
 *                exactly what SW prefetch is supposed to fix.
 *   spmv       - CSR SpMV, gather through x[col[j]]: the pass's gather path.
 *
 * Build:  clang -O2 -mcpu=oryon-1 bench.c -o bench
 *                                [-fpass-plugin=./AArch64HeteroOpt.so]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static double *xalloc(size_t n) {
  void *p = NULL;
  if (posix_memalign(&p, 64, n * sizeof(double)) != 0 || !p) {
    fprintf(stderr, "oom\n");
    exit(1);
  }
  return (double *)p;
}

/* C = A*B, loop order i,j,k */
static void dgemm_ijk(const double *restrict A, const double *restrict B,
                      double *restrict C, int N) {
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) {
      double sum = 0.0;
      for (int k = 0; k < N; k++)
        sum += A[(size_t)i * N + k] * B[(size_t)k * N + j];
      C[(size_t)i * N + j] = sum;
    }
}

/* C = A*B, loop order i,k,j: unit-stride inner loop over j */
static void dgemm_ikj(const double *restrict A, const double *restrict B,
                      double *restrict C, int N) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) C[(size_t)i * N + j] = 0.0;
    for (int k = 0; k < N; k++) {
      const double a = A[(size_t)i * N + k];
      const double *b = B + (size_t)k * N;
      double *c = C + (size_t)i * N;
      for (int j = 0; j < N; j++) c[j] += a * b[j];
    }
  }
}

/* CSR SpMV: y = A*x, 5 non-zeros per row */
static void spmv(const int *rowptr, const int *col, const double *val,
                 const double *x, double *y, int N) {
  for (int i = 0; i < N; i++) {
    double sum = 0.0;
    for (int j = rowptr[i]; j < rowptr[i + 1]; j++)
      sum += val[j] * x[col[j]];
    y[i] = sum;
  }
}

static double bench(void (*fn)(const double *, const double *, double *, int),
                    const double *A, const double *B, double *C, int N,
                    int reps) {
  double best = 1e30;
  for (int r = 0; r < reps; r++) {
    memset(C, 0, (size_t)N * N * sizeof(double));
    double t0 = now_ms();
    fn(A, B, C, N);
    double dt = now_ms() - t0;
    if (dt < best) best = dt;
  }
  return best;
}

int main(int argc, char **argv) {
  int N = (argc > 1) ? atoi(argv[1]) : 640;
  int reps = (argc > 2) ? atoi(argv[2]) : 3;
  size_t nn = (size_t)N * N;

  double *A = xalloc(nn), *B = xalloc(nn), *C = xalloc(nn);
  for (size_t i = 0; i < nn; i++) {
    A[i] = (double)((i * 1103515245u + 12345u) % 1000) / 1000.0;
    B[i] = (double)((i * 69069u + 1u) % 1000) / 1000.0;
  }

  /* CSR: 5 non-zeros per row, column indices strided by 97 to defeat
     hardware stream detection (a realistic sparse pattern). */
  int nz = N * 5;
  int *rowptr = malloc(sizeof(int) * (N + 1));
  int *col = malloc(sizeof(int) * nz);
  double *val = xalloc(nz), *x = xalloc(N), *y = xalloc(N);
  for (int i = 0; i <= N; i++) rowptr[i] = i * 5;
  for (int i = 0; i < N; i++)
    for (int j = 0; j < 5; j++) {
      col[i * 5 + j] = (i * 97 + j * 13) % N;
      val[i * 5 + j] = 1.0 / (j + 1);
    }
  for (int i = 0; i < N; i++) x[i] = 1.0 + i * 1e-3;

  double t1 = bench(dgemm_ijk, A, B, C, N, reps);
  double t2 = bench(dgemm_ikj, A, B, C, N, reps);

  double best_spmv = 1e30, sink = 0;
  const int spmv_iters = 200;   /* one SpMV is microseconds; loop it */
  for (int r = 0; r < reps; r++) {
    double t0 = now_ms();
    for (int it = 0; it < spmv_iters; it++) {
      spmv(rowptr, col, val, x, y, N);
      x[it % N] += 1e-9;        /* keep iterations distinct */
    }
    double dt = (now_ms() - t0) / spmv_iters;
    if (dt < best_spmv) best_spmv = dt;
  }
  for (int i = 0; i < N; i++) sink += y[i];

  double gflops = 2.0 * (double)N * N * N / (t1 * 1e6);
  printf("N=%-5d dgemm_ijk %8.2f ms   dgemm_ikj %8.2f ms   spmv %7.3f ms   "
         "[%.2f GFLOP/s]  chk=%.6g\n",
         N, t1, t2, best_spmv, gflops, (t1 + t2) / 1e9 + sink * 1e-18);

  free(A); free(B); free(C); free(rowptr); free(col); free(val);
  free(x); free(y);
  return 0;
}
