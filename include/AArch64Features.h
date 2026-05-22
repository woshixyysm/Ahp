#pragma once

#include <cstdint>
#include <string_view>

/// Ahp – AArch64 Hetero-Opt Pass  (production build)
/// This header is C++-only; no LLVM dependency.

namespace aarch64hetero {

// ---------------------------------------------------------------------------
// Version stamp – bump on ABI-breaking changes.
// ---------------------------------------------------------------------------
inline constexpr uint32_t AHP_VERSION_MAJOR = 0;
inline constexpr uint32_t AHP_VERSION_MINOR = 4;
inline constexpr uint32_t AHP_VERSION_PATCH = 0;
inline constexpr uint32_t AHP_VERSION =
    (AHP_VERSION_MAJOR << 16) | (AHP_VERSION_MINOR << 8) | AHP_VERSION_PATCH;

// ---------------------------------------------------------------------------
// Processor families relevant to hetero-opt.
// ---------------------------------------------------------------------------
enum class HeteroProcFamily : uint8_t {
  Unknown,
  Ampere1,       ///< AmpereOne (AC03)
  Ampere1A,      ///< AmpereOne AC04+
  Oryon,         ///< Qualcomm Oryon (Snapdragon X / 8 Elite P-core)
  NeoverseN1,    ///< Arm Neoverse N1
  NeoverseN2,    ///< Arm Neoverse N2
  NeoverseV1,    ///< Arm Neoverse V1
  CortexX4,      ///< Arm Cortex-X4
  AppleM4,       ///< Apple M4 (structural similarity to Oryon)
};

/// Map LLVM -mcpu string to hetero family (pure C++, no LLVM dependency).
///
/// Matching rules (longest-prefix-wins via ordering):
///   "ampere1a"  → Ampere1A   (must precede "ampere1" check)
///   "ampere1"   → Ampere1
///   "ampere"    → Ampere1    (generic ampere brand)
///   "oryon"     → Oryon      (matches "oryon-1", "oryon-2", ...)
///   "neoverse-n1" → NeoverseN1  (exact)
///   "neoverse-n2" → NeoverseN2  (exact)
///   "neoverse-v1" → NeoverseV1  (exact)
///   "cortex-x4" → CortexX4   (prefix)
///   "apple-m4"  → AppleM4    (exact)
[[nodiscard]]
inline HeteroProcFamily cpuToFamily(std::string_view cpu) noexcept {
  // Ampere variants — order matters: longest prefix first.
  if (cpu.starts_with("ampere1a"))           return HeteroProcFamily::Ampere1A;
  if (cpu.starts_with("ampere1"))            return HeteroProcFamily::Ampere1;
  if (cpu.starts_with("ampere"))             return HeteroProcFamily::Ampere1;

  // Qualcomm Oryon — prefix match covers oryon-1, oryon-2, etc.
  if (cpu.starts_with("oryon"))              return HeteroProcFamily::Oryon;

  // Arm Neoverse — exact match to avoid false hits on future variants.
  if (cpu == "neoverse-n1")                  return HeteroProcFamily::NeoverseN1;
  if (cpu == "neoverse-n2")                  return HeteroProcFamily::NeoverseN2;
  if (cpu == "neoverse-v1")                  return HeteroProcFamily::NeoverseV1;

  // Cortex / Apple — prefix / exact.
  if (cpu.starts_with("cortex-x4"))         return HeteroProcFamily::CortexX4;
  if (cpu == "apple-m4")                     return HeteroProcFamily::AppleM4;

  return HeteroProcFamily::Unknown;
}

/// Return a human-readable name for a processor family (for diagnostics).
[[nodiscard]]
inline std::string_view familyName(HeteroProcFamily f) noexcept {
  switch (f) {
  case HeteroProcFamily::Ampere1:     return "AmpereOne";
  case HeteroProcFamily::Ampere1A:    return "AmpereOne-A";
  case HeteroProcFamily::Oryon:       return "Qualcomm-Oryon";
  case HeteroProcFamily::NeoverseN1:  return "Neoverse-N1";
  case HeteroProcFamily::NeoverseN2:  return "Neoverse-N2";
  case HeteroProcFamily::NeoverseV1:  return "Neoverse-V1";
  case HeteroProcFamily::CortexX4:    return "Cortex-X4";
  case HeteroProcFamily::AppleM4:     return "Apple-M4";
  case HeteroProcFamily::Unknown:     return "Unknown";
  }
  return "Unknown"; // unreachable, silences -Wreturn-type
}

} // namespace aarch64hetero
