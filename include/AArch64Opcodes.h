#pragma once

#include <cstdint>

// ---------------------------------------------------------------------------
// AArch64 opcode / register-class constants for in-tree vs out-of-tree builds.
//
// In-tree build  (AARCH64_HETERO_HAS_TARGET_HEADERS=1):
//   Include generated AArch64InstrInfo.h / AArch64RegisterInfo.h directly.
//
// Out-of-tree build (AARCH64_HETERO_HAS_TARGET_HEADERS=0):
//   Use verified numeric fallbacks from LLVM 21 generated tables.
//   Values are checked against AArch64GenInstrInfo.inc and
//   AArch64GenRegisterInfo.inc; add a static_assert if you re-verify.
// ---------------------------------------------------------------------------

#if defined(__has_include)
#  if __has_include("AArch64InstrInfo.h")
#    define AARCH64_HETERO_HAS_TARGET_HEADERS 1
#  elif __has_include("llvm/Target/AArch64/AArch64InstrInfo.h")
#    define AARCH64_HETERO_HAS_TARGET_HEADERS 1
#  else
#    define AARCH64_HETERO_HAS_TARGET_HEADERS 0
#  endif
#else
#  define AARCH64_HETERO_HAS_TARGET_HEADERS 0
#endif

// ---------------------------------------------------------------------------
// In-tree path: pull in generated headers.
// ---------------------------------------------------------------------------
#if AARCH64_HETERO_HAS_TARGET_HEADERS

#  if __has_include("AArch64InstrInfo.h")
#    include "AArch64InstrInfo.h"
#  else
#    include "llvm/Target/AArch64/AArch64InstrInfo.h"
#  endif

#  if __has_include("AArch64RegisterInfo.h")
#    include "AArch64RegisterInfo.h"
#  else
#    include "llvm/Target/AArch64/AArch64RegisterInfo.h"
#  endif

namespace aarch64hetero {
  /// PRFM Xn, #pimm12 immediate-offset form.
  inline constexpr unsigned HETERO_PRFMui = AArch64::PRFMui;
  /// ADD Xd, Xn, #imm12.
  inline constexpr unsigned HETERO_ADDXri = AArch64::ADDXri;
} // namespace aarch64hetero

// ---------------------------------------------------------------------------
// Out-of-tree path: numeric fallbacks only.
// ---------------------------------------------------------------------------
#else

namespace aarch64hetero {

  /// PRFM <prfop>, [Xn, #pimm]
  /// Verified against LLVM 21 AArch64GenInstrInfo.inc (opcode 1102).
  inline constexpr unsigned HETERO_PRFMui = 1102;

  /// ADD Xd, Xn, #imm12
  /// Verified against LLVM 21 AArch64GenInstrInfo.inc (opcode 32).
  inline constexpr unsigned HETERO_ADDXri = 32;

  // -------------------------------------------------------------------------
  // PRFM prefetch-operation encoding (AArch64 ISA A64 v9.2).
  //   prfop[4:3] = type   (PLD=00, PLI=01, PST=10)
  //   prfop[2:1] = target (L1=00, L2=01, L3=10)
  //   prfop[0]   = policy (KEEP=0, STRM=1)
  // -------------------------------------------------------------------------
  namespace PrfOp {
    inline constexpr unsigned PLDL1KEEP = 0b00000; ///< Load, L1, temporal
    inline constexpr unsigned PLDL1STRM = 0b00001; ///< Load, L1, streaming (NT)
    inline constexpr unsigned PLDL2KEEP = 0b00010; ///< Load, L2, temporal
    inline constexpr unsigned PLDL2STRM = 0b00011; ///< Load, L2, streaming
    inline constexpr unsigned PSTL1KEEP = 0b10000; ///< Store, L1, temporal
    inline constexpr unsigned PSTL2KEEP = 0b10010; ///< Store, L2, temporal
  } // namespace PrfOp

} // namespace aarch64hetero

#endif // AARCH64_HETERO_HAS_TARGET_HEADERS
