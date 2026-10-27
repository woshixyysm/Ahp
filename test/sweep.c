/* Stride sweep: same shape, only the induction step changes.
   Bytes per iteration = step * 8 (double). */
void s1(double *restrict o, const double *restrict i, int n) {   /* 8 B  */
  for (int k = 0; k < n; k += 1) o[k] = i[k] * 2.0;
}
void s2(double *restrict o, const double *restrict i, int n) {   /* 16 B */
  for (int k = 0; k < n; k += 2) o[k] = i[k] * 2.0;
}
void s8(double *restrict o, const double *restrict i, int n) {   /* 64 B */
  for (int k = 0; k < n; k += 8) o[k] = i[k] * 2.0;
}
void s9(double *restrict o, const double *restrict i, int n) {   /* 72 B */
  for (int k = 0; k < n; k += 9) o[k] = i[k] * 2.0;
}
void s16(double *restrict o, const double *restrict i, int n) {  /* 128 B */
  for (int k = 0; k < n; k += 16) o[k] = i[k] * 2.0;
}
void s65(double *restrict o, const double *restrict i, int n) {  /* 520 B */
  for (int k = 0; k < n; k += 65) o[k] = i[k] * 2.0;
}
