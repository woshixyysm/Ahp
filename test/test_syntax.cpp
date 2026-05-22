//===- test_syntax.cpp - Header inclusion and instantiation smoke test ----===//
//
// Verifies that all Ahp pure-C++ headers compile and that detectProfile()
// works correctly.
//
// Two build modes:
//   A) With real LLVM headers (HAS_LLVM_HEADERS=1, from CMake LLVM build):
//      Uses the real llvm::StringRef. All headers compile normally.
//
//   B) Without LLVM (standalone Makefile build):
//      Provides a minimal llvm::StringRef stub and enables
//      AARCH64_HETERO_HAS_REAL_LLVM=1 so detectProfile() is compiled.
//
//===----------------------------------------------------------------------===//

// ---------------------------------------------------------------------------
// Mode B: standalone build – provide StringRef stub before any Ahp headers.
// ---------------------------------------------------------------------------
#if !defined(HAS_LLVM_HEADERS) || !HAS_LLVM_HEADERS

// These macros must be set before including AArch64TargetProfileLLVM.h so
// that the header uses our stub instead of trying to find llvm/ADT/StringRef.h.
#define AARCH64_HETERO_HAS_REAL_LLVM 1

#include <cstddef>
#include <string_view>

namespace llvm {

class StringRef {
  const char *data_;
  size_t      len_;
public:
  constexpr StringRef() noexcept : data_(nullptr), len_(0) {}
  StringRef(const char *s) noexcept
      : data_(s), len_(s ? __builtin_strlen(s) : 0) {}
  StringRef(const char *s, size_t n) noexcept : data_(s), len_(n) {}

  [[nodiscard]] bool starts_with(std::string_view prefix) const noexcept {
    if (len_ < prefix.size()) return false;
    return std::string_view(data_, len_).substr(0, prefix.size()) == prefix;
  }
  [[nodiscard]] bool starts_with(const char *prefix) const noexcept {
    return starts_with(std::string_view(prefix));
  }
  [[nodiscard]] bool contains(std::string_view s) const noexcept {
    return std::string_view(data_, len_).find(s) != std::string_view::npos;
  }
  [[nodiscard]] bool empty() const noexcept { return len_ == 0; }
  [[nodiscard]] size_t size() const noexcept { return len_; }
  [[nodiscard]] const char *data() const noexcept { return data_; }
  [[nodiscard]] operator std::string_view() const noexcept {
    return {data_, len_};
  }
};

} // namespace llvm

// Include Ahp headers in stub mode (no real LLVM IR types).
#include "AArch64TargetProfile.h"
#include "AArch64TargetProfileLLVM.h"
#include "AArch64Opcodes.h"
#include "AArch64Features.h"
// AArch64HeteroOpt.h and AArch64PrefetchAnalysis.h require real LLVM IR
// types; skip them in stub mode.

// ---------------------------------------------------------------------------
// Mode A: real LLVM headers available – include everything normally.
// ---------------------------------------------------------------------------
#else // HAS_LLVM_HEADERS

#include "llvm/ADT/StringRef.h"
#include "AArch64TargetProfile.h"
#include "AArch64TargetProfileLLVM.h"
#include "AArch64Opcodes.h"
#include "AArch64Features.h"
#include "AArch64HeteroOpt.h"
// AArch64PrefetchAnalysis.h included transitively.

#endif // HAS_LLVM_HEADERS

#include <cassert>
#include <cstdio>

// ---------------------------------------------------------------------------
// Test utilities
// ---------------------------------------------------------------------------
static int failures = 0;

#define CHECK(expr, msg)                                                  \
  do {                                                                    \
    if (!(expr)) {                                                         \
      fprintf(stderr, "[FAIL] %s  (%s:%d)\n", msg, __FILE__, __LINE__);   \
      ++failures;                                                          \
    } else {                                                               \
      printf("[PASS] %s\n", msg);                                          \
    }                                                                      \
  } while (0)

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
static void test_factory_instantiation() {
  auto p1 = aarch64hetero::makeAmpereAltraProfile();
  auto p2 = aarch64hetero::makeSnapdragon8EliteProfile();
  CHECK(p1.kind == aarch64hetero::TargetKind::AmpereAltra,    "Ampere factory");
  CHECK(p2.kind == aarch64hetero::TargetKind::SnapdragonElite,"Oryon factory");
  (void)p1; (void)p2;
}

static void test_detect_profile() {
  using namespace aarch64hetero;

  auto oryon  = detectProfile(llvm::StringRef("oryon-1"));
  auto oryon2 = detectProfile(llvm::StringRef("oryon-2"));
  auto ampere = detectProfile(llvm::StringRef("ampere1"));
  auto neon1  = detectProfile(llvm::StringRef("neoverse-n1"));
  auto unk    = detectProfile(llvm::StringRef("generic"));
  auto empty  = detectProfile(llvm::StringRef(""));

  CHECK(oryon.kind  == TargetKind::SnapdragonElite,  "detect oryon-1 → Oryon");
  CHECK(oryon2.kind == TargetKind::SnapdragonElite,  "detect oryon-2 → Oryon");
  CHECK(ampere.kind == TargetKind::AmpereAltra,      "detect ampere1 → Ampere");
  CHECK(neon1.kind  == TargetKind::AmpereAltra,      "detect neoverse-n1 → Ampere");

  // Unknown CPU must NEVER silently enable SW prefetch (safety contract).
  CHECK(unk.inject_sw_prefetch   == false, "unknown CPU: SW prefetch disabled");
  CHECK(empty.inject_sw_prefetch == false, "empty CPU: SW prefetch disabled");

  // Oryon must enable SW prefetch and SVE guard.
  CHECK(oryon.inject_sw_prefetch  == true,  "Oryon: SW prefetch enabled");
  CHECK(oryon.sve_guard == aarch64hetero::SveGuardMode::InnerOnly,  "Oryon: SVE guard enabled");

  // Ampere must not enable SVE guard (no SVE2 tail-folding).
  CHECK(ampere.sve_guard == aarch64hetero::SveGuardMode::Disabled, "Ampere: SVE guard disabled");
}

static void test_opcodes_present() {
  CHECK(aarch64hetero::HETERO_PRFMui == 1102u, "HETERO_PRFMui=1102");
  CHECK(aarch64hetero::HETERO_ADDXri == 32u,   "HETERO_ADDXri=32");
  CHECK(aarch64hetero::PrfOp::PLDL1KEEP == 0b00000u, "PLDL1KEEP=0");
  CHECK(aarch64hetero::PrfOp::PLDL1STRM == 0b00001u, "PLDL1STRM=1");
  CHECK(aarch64hetero::PrfOp::PLDL2KEEP == 0b00010u, "PLDL2KEEP=2");
  CHECK(aarch64hetero::PrfOp::PLDL2STRM == 0b00011u, "PLDL2STRM=3");
}

int main() {
  printf("=== Syntax / instantiation smoke test ===\n\n");
  test_factory_instantiation();
  test_detect_profile();
  test_opcodes_present();
  printf("\n%s\n", failures == 0 ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
  return failures == 0 ? 0 : 1;
}
