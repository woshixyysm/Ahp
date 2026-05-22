#pragma once

#include "AArch64TargetProfile.h"

#ifdef HAS_LLVM_HEADERS

#include "llvm/IR/PassManager.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/Statistic.h"

// NOTE: DEBUG_TYPE is intentionally NOT defined here.
// It is defined only in AArch64HeteroOpt.cpp (after all LLVM headers are
// included) to avoid redefinition conflicts with LLVM's own headers that
// define DEBUG_TYPE internally (e.g. CGSCCPassManager.h defines "cgscc").

namespace aarch64hetero {

/// AArch64HeteroOptPass
///
/// New-PM function pass that injects llvm.prefetch intrinsics for
/// memory-bound loops on Oryon and server-class AArch64 targets.
///
/// Injection policy summary:
///   - Only runs when TargetProfile::inject_sw_prefetch == true.
///   - Skips vectorized loops (HW prefetcher handles those well).
///   - Skips loops whose pointer provenance is not proven heap/global.
///   - Guards against SVE tail-folding livelock: when guard_sve_tail_fold
///     is true AND ScalarEvolution cannot establish a constant backedge
///     taken count, the loop is skipped entirely.
///   - Respects PrefetchBudget (per-function and per-loop caps).
///   - Uses target-specific locality values from localityForAccess().
class AArch64HeteroOptPass
    : public llvm::PassInfoMixin<AArch64HeteroOptPass> {
public:
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &FAM);
};

} // namespace aarch64hetero

#endif // HAS_LLVM_HEADERS
