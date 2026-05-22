#pragma once

#include "AArch64TargetProfile.h"
#include "AArch64Features.h"

// ---------------------------------------------------------------------------
// Auto-detect LLVM availability unless the caller has pre-defined the macro.
// Pre-defining AARCH64_HETERO_HAS_REAL_LLVM=1 before including this header
// (e.g., in tests with a minimal StringRef stub) overrides auto-detection.
// ---------------------------------------------------------------------------
#if !defined(AARCH64_HETERO_HAS_REAL_LLVM)
#  if defined(__has_include) && __has_include("llvm/ADT/StringRef.h")
#    define AARCH64_HETERO_HAS_REAL_LLVM 1
#  else
#    define AARCH64_HETERO_HAS_REAL_LLVM 0
#  endif
#endif

#if AARCH64_HETERO_HAS_REAL_LLVM

// In real LLVM builds, include the real header.
// In stub-based tests, llvm::StringRef is already declared by the test file.
#if defined(__has_include) && __has_include("llvm/ADT/StringRef.h")
#  include "llvm/ADT/StringRef.h"
#endif

namespace aarch64hetero {

/// Detect target profile from LLVM -mcpu string.
///
/// Mapping semantics:
///   Ampere / Neoverse server-class → AmpereAltra profile
///     (SW prefetch disabled; strong HW prefetcher handles streaming)
///   Oryon / Cortex-X4 / Apple-M4  → SnapdragonElite profile
///     (SW prefetch enabled; P-core focus; SVE tail-fold guard active)
///   Unknown CPU                    → AmpereAltra profile as conservative
///     server-side default (inject_sw_prefetch=false, safe).
///
/// NOTE: We never silently enable SW prefetch on unknown targets.
/// Injecting prefetch on a CPU whose micro-architecture we don't know
/// can degrade performance and waste AGU bandwidth.
[[nodiscard]]
inline TargetProfile detectProfile(llvm::StringRef cpu) noexcept {
  const HeteroProcFamily fam = cpuToFamily(
      std::string_view(cpu.data(), cpu.size()));

  switch (fam) {
  // Ampere / Neoverse: homogeneous server, strong HW prefetcher.
  case HeteroProcFamily::Ampere1:
  case HeteroProcFamily::Ampere1A:
  case HeteroProcFamily::NeoverseN1:
  case HeteroProcFamily::NeoverseN2:
  case HeteroProcFamily::NeoverseV1:
    return makeAmpereAltraProfile();

  // Oryon / Cortex-X4 / Apple-M4: high-IPC big core with SME/SVE2.
  case HeteroProcFamily::Oryon:
  case HeteroProcFamily::CortexX4:
  case HeteroProcFamily::AppleM4:
    return makeSnapdragon8EliteProfile();

  // Unknown: use conservative server profile (no SW prefetch).
  case HeteroProcFamily::Unknown:
    return makeAmpereAltraProfile();
  }

  // Unreachable: all enum values handled. Satisfy -Wreturn-type.
  return makeAmpereAltraProfile();
}

} // namespace aarch64hetero

#endif // AARCH64_HETERO_HAS_REAL_LLVM
