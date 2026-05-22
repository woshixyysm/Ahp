#pragma once

#ifdef HAS_LLVM_HEADERS

#include "AArch64TargetProfile.h"

#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/LoopUtils.h"

// NOTE: DEBUG_TYPE intentionally not defined here; define it in the .cpp
// translation unit that uses this header, after all LLVM headers.

namespace aarch64hetero {

// ---------------------------------------------------------------------------
// Access-pattern classification for prefetch decisions.
// ---------------------------------------------------------------------------
enum class AccessPattern {
  Sequential, ///< Affine AddRec with loop-invariant step (stride known)
  Strided,    ///< Non-affine AddRec or loop-variant step
  Indirect,   ///< Not an AddRec – pointer chasing / index from memory
  Unknown,    ///< Loop-invariant pointer (no memory iteration)
};

/// Classify a pointer SCEV expression relative to a loop.
[[nodiscard]]
inline AccessPattern classifyAccessPattern(const llvm::SCEV *PtrSCEV,
                                           llvm::ScalarEvolution &SE,
                                           const llvm::Loop *L) noexcept {
  using namespace llvm;
  if (SE.isLoopInvariant(PtrSCEV, L))
    return AccessPattern::Unknown;

  if (const auto *AR = dyn_cast<SCEVAddRecExpr>(PtrSCEV)) {
    if (AR->isAffine() && SE.isLoopInvariant(AR->getStepRecurrence(SE), L))
      return AccessPattern::Sequential;
    return AccessPattern::Strided;
  }
  return AccessPattern::Indirect;
}

// ---------------------------------------------------------------------------
// AArch64PrefetchInjector – loop-level pass for indirect pointer chasing.
//
// Rationale: HW prefetchers on both Ampere and Oryon handle sequential /
// strided streams well.  They are blind to pointer-chasing patterns
// (linked-list, tree traversal, hash-table probe).  This pass targets only
// those indirect loads of pointer type.
//
// Strategy:
//   For each indirect load of pointer type inside an innermost loop, insert
//   a prefetch of the *loaded* pointer (one level of indirection ahead).
//   This implements the classic "prefetch next node" technique:
//
//     p = p->next;           // load of pointer
//     prefetch(p);           // prefetch next->next before we get there
//     use(p->val);           // this access now hits cache
//
// Locality selection:
//   Oryon  → locality 1 (PLDL1STRM): streaming, avoid L1 pollution
//   Ampere → locality 2 (PLDL2KEEP): hold in L2, don't thrash L1
//
// SVE tail-folding guard:
//   When profile.guard_sve_tail_fold is set, loops without a proven
//   constant backedge taken count are skipped to avoid the LLVM
//   LoopVectorizer tail-folding livelock (see README §问题1).
// ---------------------------------------------------------------------------
class AArch64PrefetchInjector
    : public llvm::PassInfoMixin<AArch64PrefetchInjector> {
public:
  explicit AArch64PrefetchInjector(TargetProfile P)
      : Profile_(std::move(P)) {}

  llvm::PreservedAnalyses
  run(llvm::Loop &L, llvm::LoopAnalysisManager &LAM,
      llvm::LoopStandardAnalysisResults &AR, llvm::LPMUpdater &) {
    using namespace llvm;

    if (!Profile_.inject_sw_prefetch) return PreservedAnalyses::all();
    if (!L.getSubLoops().empty())     return PreservedAnalyses::all();

    ScalarEvolution &SE = AR.SE;

    // SVE tail-folding guard (see README §问题1).
    if (Profile_.guard_sve_tail_fold) {
      const SCEV *BTC = SE.getBackedgeTakenCount(&L);
      if (isa<SCEVCouldNotCompute>(BTC))
        return PreservedAnalyses::all();
    }

    bool Changed = false;
    Module      *M   = L.getHeader()->getParent()->getParent();
    LLVMContext &Ctx = M->getContext();

    // Locality: PLDL1STRM (1) for Oryon, PLDL2KEEP (2) for Ampere.
    const uint32_t Locality =
        (Profile_.kind == TargetKind::SnapdragonElite) ? 1u : 2u;

    Type *I8PtrTy = PointerType::get(Type::getInt8Ty(Ctx), 0);
    FunctionCallee PrefetchFn = Intrinsic::getOrInsertDeclaration(
        M, Intrinsic::prefetch, {I8PtrTy});

    uint32_t LoopBudget = Profile_.budget.per_loop;

    for (BasicBlock *BB : L.blocks()) {
      for (Instruction &I : *BB) {
        if (LoopBudget == 0) break;

        auto *LI = dyn_cast<LoadInst>(&I);
        if (!LI || !LI->isSimple()) continue;
        if (!LI->getType()->isPointerTy()) continue;

        const SCEV *PtrSCEV = SE.getSCEV(LI->getPointerOperand());
        if (classifyAccessPattern(PtrSCEV, SE, &L) != AccessPattern::Indirect)
          continue;

        // Insert prefetches BEFORE the load (pointer operand is live here).
        //
        // Classic two-hop pointer-chasing software pipeline:
        //
        //   Iteration N:  p = load(cur)          ← accesses cur node
        //   Iteration N+1: p = load(cur->next)   ← accesses next node
        //
        // Prefetch strategy (inserted BEFORE current load):
        //   Hop-1: prfm [cur]          → warms the node we're about to read
        //   Hop-2: tmp = load [cur];   → speculative: read cur->next ptr
        //          prfm [tmp]           → warms cur->next->next (2 hops ahead)
        //
        // Inserting before the load is always safe: the pointer operand
        // dominates its own load.  The speculative hop-2 load is read-only
        // and cannot trap (same pointer type, naturally aligned).
        IRBuilder<> B(LI);  // insert point = before the load

        // Hop-1: warm the node we are loading FROM
        Value *CurNodeCast = B.CreatePointerCast(
            LI->getPointerOperand(), I8PtrTy, "pf.hop1");
        B.CreateCall(PrefetchFn, {
          CurNodeCast,
          B.getInt32(0),
          B.getInt32(Locality),
          B.getInt32(1)
        });
        --LoopBudget;
        Changed = true;

        // Hop-2: speculatively load next-pointer and warm two hops ahead
        if (LoopBudget > 0) {
          Value *NextNodePtr = B.CreateLoad(
              LI->getType(), LI->getPointerOperand(), "pf.hop2.ptr");
          Value *NextNodeCast = B.CreatePointerCast(
              NextNodePtr, I8PtrTy, "pf.hop2");
          B.CreateCall(PrefetchFn, {
            NextNodeCast,
            B.getInt32(0),
            B.getInt32(Locality),
            B.getInt32(1)
          });
          --LoopBudget;
        }
      }
      if (LoopBudget == 0) break;
    }

    if (!Changed) return PreservedAnalyses::all();
    PreservedAnalyses PA;
    PA.preserveSet<CFGAnalyses>();
    return PA;
  }

private:
  TargetProfile Profile_;
};

/// Register AArch64PrefetchInjector with a PassBuilder at LoopOptimizerEnd.
inline void registerPrefetchInjector(llvm::PassBuilder &PB,
                                     const TargetProfile &Profile) {
  PB.registerLoopOptimizerEndEPCallback(
    [Profile](llvm::LoopPassManager &LPM, llvm::OptimizationLevel Opt) {
      if (Opt.isOptimizingForSpeed())
        LPM.addPass(AArch64PrefetchInjector(Profile));
    });
}

} // namespace aarch64hetero

#endif // HAS_LLVM_HEADERS
