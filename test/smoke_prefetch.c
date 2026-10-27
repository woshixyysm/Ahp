/* Ahp smoke test: 128-byte stride, dynamic trip count.
   Expected on -mcpu=oryon-1: ModReuse -> locality 3, one or more
   llvm.prefetch calls. */
void strided(double *restrict out, const double *restrict in, int n) {
  for (int i = 0; i < n; i += 16)
    out[i] = in[i] * 2.0;
}

/* unit stride, innermost, no outer loop -> Streaming, locality 1 on Oryon */
void stream(float *restrict out, const float *restrict in, int n) {
  for (int i = 0; i < n; i++)
    out[i] = in[i] + 1.0f;
}
