//===- test_features.cpp - Tests for AArch64Features.h --------------------===//
//
// Tests for cpuToFamily() and familyName() in AArch64Features.h.
// No LLVM dependency required.
//
//===----------------------------------------------------------------------===//

#include "AArch64Features.h"
#include <cassert>
#include <cstdio>
#include <string_view>

using namespace aarch64hetero;

// ---------------------------------------------------------------------------
// Helper: run one assertion and report pass/fail
// ---------------------------------------------------------------------------
static int failures = 0;

#define CHECK(expr, msg)                                          \
  do {                                                           \
    if (!(expr)) {                                               \
      fprintf(stderr, "[FAIL] %s  (%s:%d)\n", msg, __FILE__, __LINE__); \
      ++failures;                                                \
    } else {                                                     \
      printf("[PASS] %s\n", msg);                                \
    }                                                            \
  } while (0)

// ---------------------------------------------------------------------------
// cpuToFamily – exact matches
// ---------------------------------------------------------------------------
static void test_exact_matches() {
  CHECK(cpuToFamily("neoverse-n1") == HeteroProcFamily::NeoverseN1,
        "neoverse-n1 → NeoverseN1");
  CHECK(cpuToFamily("neoverse-n2") == HeteroProcFamily::NeoverseN2,
        "neoverse-n2 → NeoverseN2");
  CHECK(cpuToFamily("neoverse-v1") == HeteroProcFamily::NeoverseV1,
        "neoverse-v1 → NeoverseV1");
  CHECK(cpuToFamily("apple-m4")    == HeteroProcFamily::AppleM4,
        "apple-m4 → AppleM4");
  CHECK(cpuToFamily("unknown")     == HeteroProcFamily::Unknown,
        "unknown → Unknown");
  CHECK(cpuToFamily("")            == HeteroProcFamily::Unknown,
        "empty string → Unknown");
}

// ---------------------------------------------------------------------------
// cpuToFamily – prefix matches (Ampere)
// ---------------------------------------------------------------------------
static void test_ampere_prefix() {
  // Longest-prefix-first ordering: ampere1a beats ampere1.
  CHECK(cpuToFamily("ampere1a")    == HeteroProcFamily::Ampere1A, "ampere1a → Ampere1A");
  CHECK(cpuToFamily("ampere1a-x")  == HeteroProcFamily::Ampere1A, "ampere1a-x → Ampere1A");
  CHECK(cpuToFamily("ampere1")     == HeteroProcFamily::Ampere1,  "ampere1 → Ampere1");
  CHECK(cpuToFamily("ampere1-foo") == HeteroProcFamily::Ampere1,  "ampere1-foo → Ampere1");
  CHECK(cpuToFamily("ampere")      == HeteroProcFamily::Ampere1,  "ampere (generic) → Ampere1");
}

// ---------------------------------------------------------------------------
// cpuToFamily – prefix matches (Oryon)
// ---------------------------------------------------------------------------
static void test_oryon_prefix() {
  // Critical: "oryon-1" must map to Oryon (was broken in original code).
  CHECK(cpuToFamily("oryon")   == HeteroProcFamily::Oryon, "oryon → Oryon");
  CHECK(cpuToFamily("oryon-1") == HeteroProcFamily::Oryon, "oryon-1 → Oryon");
  CHECK(cpuToFamily("oryon-2") == HeteroProcFamily::Oryon, "oryon-2 → Oryon");
}

// ---------------------------------------------------------------------------
// cpuToFamily – Cortex-X4 prefix
// ---------------------------------------------------------------------------
static void test_cortex_x4() {
  CHECK(cpuToFamily("cortex-x4")   == HeteroProcFamily::CortexX4, "cortex-x4 → CortexX4");
  CHECK(cpuToFamily("cortex-x4-a") == HeteroProcFamily::CortexX4, "cortex-x4-a → CortexX4");
}

// ---------------------------------------------------------------------------
// familyName – round-trip sanity
// ---------------------------------------------------------------------------
static void test_family_name() {
  CHECK(familyName(HeteroProcFamily::Ampere1)    == "AmpereOne",       "familyName Ampere1");
  CHECK(familyName(HeteroProcFamily::Ampere1A)   == "AmpereOne-A",     "familyName Ampere1A");
  CHECK(familyName(HeteroProcFamily::Oryon)      == "Qualcomm-Oryon",  "familyName Oryon");
  CHECK(familyName(HeteroProcFamily::NeoverseN1) == "Neoverse-N1",     "familyName NeoverseN1");
  CHECK(familyName(HeteroProcFamily::NeoverseN2) == "Neoverse-N2",     "familyName NeoverseN2");
  CHECK(familyName(HeteroProcFamily::NeoverseV1) == "Neoverse-V1",     "familyName NeoverseV1");
  CHECK(familyName(HeteroProcFamily::CortexX4)   == "Cortex-X4",       "familyName CortexX4");
  CHECK(familyName(HeteroProcFamily::AppleM4)    == "Apple-M4",        "familyName AppleM4");
  CHECK(familyName(HeteroProcFamily::Unknown)    == "Unknown",          "familyName Unknown");
}

// ---------------------------------------------------------------------------
// Version stamp present
// ---------------------------------------------------------------------------
static void test_version() {
  CHECK(AHP_VERSION_MAJOR == 0, "version major is 0");
  CHECK(AHP_VERSION_MINOR == 4, "version minor is 4");
  CHECK(AHP_VERSION       == (0u << 16 | 4u << 8 | 0u), "version packed");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
  printf("=== AArch64Features tests ===\n\n");
  test_exact_matches();
  test_ampere_prefix();
  test_oryon_prefix();
  test_cortex_x4();
  test_family_name();
  test_version();
  printf("\n%s\n", failures == 0 ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
  return failures == 0 ? 0 : 1;
}
