//===- AArch64HeteroOpt.cpp - AArch64 heterogeneous prefetch pass ---------===//
//
// Part of the Ahp project (AArch64 Hetero-Opt Pass).
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ── Architecture (v0.7) ───────────────────────────────────────────────────
//
// Two passes, two extension points:
//
//  [LoopOptimizerEndEP] AArch64LaneHintPass (LoopPass, pre-vectorizer)
//    Injects llvm.loop.unroll.count + llvm.loop.interleave.count on FP
//    reduction loops. setLoopID is safe here (post loop-opts, pre-vector).
//
//  [OptimizerLastEP] AArch64HeteroOptPass (FunctionPass, post-vectorizer)
//    Injects llvm.prefetch only. NO setLoopID. NO new loads.
//
// ── Critical correctness rule for prefetch insertion point ────────────────
//
//  The prefetch address must be computed from values that DOMINATE the
//  insertion point. There are two cases:
//
//  Case A – Loop-invariant base pointer (e.g., function argument %p):
//    prefetch_addr = GEP i8, %p, (lookahead * stride)
//    Insert in: loop PREHEADER. This prefetches once before the loop,
//    warming the cache for the first iteration's memory window.
//    ✓ Dominates all uses: %p is defined before the loop.
//
//  Case B – Loop-variant pointer (e.g., vectorized GEP %gep = gep %p, %iv):
//    The pointer value is defined INSIDE the loop body.
//    We CANNOT insert its GEP in the preheader (dominance violation → crash).
//    Insert in: SAME BB as the load, immediately BEFORE the load.
//    The prefetch covers the next iteration's data (lookahead applied inline).
//    ✓ Dominates: the prefetch is inserted at load's position, values are live.
//
//  Case B is what happens after vectorization: the vectorizer creates loop-
//  variant GEPs for each lane. Our preheader approach was always wrong for
//  vectorized loops (but we correctly skip those via isAnyVectorized).
//  For scalar loops post-vectorizer, the induction variable's GEP IS a
//  loop-variant AddRec value, so we must insert inline.
//
//  Safe insertion rule used in this version:
//    ALWAYS insert immediately BEFORE the load (inline in the loop body).
//    This is always correct regardless of whether the pointer is invariant.
//    The preheader optimization is NOT used (too risky without dominance check).
//
//===----------------------------------------------------------------------===//

#include "AArch64HeteroOpt.h"
#include "AArch64TargetProfileLLVM.h"

#include <climits>   // INT64_MAX, UINT64_MAX
#include <algorithm> // std::clamp, std::max, std::min

#ifdef HAS_LLVM_HEADERS

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"

#ifdef DEBUG_TYPE
#  undef DEBUG_TYPE
#endif
#define DEBUG_TYPE "aarch64-hetero-opt"

using namespace llvm;
using namespace aarch64hetero;

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------
STATISTIC(NumStridedPFInjected,     "Strided llvm.prefetch injected");
STATISTIC(NumGatherPFInjected,      "Gather llvm.prefetch injected");
STATISTIC(NumLoopsOptimized,        "Loops with prefetch injected");
STATISTIC(NumLoopsSkippedVec,       "Loops skipped: vectorized");
STATISTIC(NumLoopsSkippedSVE,       "Loops skipped: SVE inner + dynamic TC");
STATISTIC(NumLoopsSkippedBudget,    "Loops skipped: budget exhausted");
STATISTIC(NumLoadsSkippedInvariant, "Loads skipped: loop-invariant");
STATISTIC(NumLoadsSkippedStreaming, "Loads skipped: streaming/Ampere HW PF");
STATISTIC(NumGatherLoopsDetected,   "Gather/SpMV pattern detected");
STATISTIC(NumLaneHintsInjected,     "Loops with independent-lane hints");

// ==========================================================================
// §1  Target detection
// ==========================================================================
static TargetProfile detectTarget(const Function &F) {
  StringRef CPU = "generic";
  if (F.hasFnAttribute("target-cpu"))
    CPU = F.getFnAttribute("target-cpu").getValueAsString();
  if (F.hasFnAttribute("target-features")) {
    StringRef Feats = F.getFnAttribute("target-features").getValueAsString();
    if (Feats.contains("+oryon"))        CPU = "oryon-1";
    else if (Feats.contains("+ampere1")) CPU = "ampere1";
  }
  return detectProfile(CPU);
}

// ==========================================================================
// §2  Loop classification
// ==========================================================================

static bool isScalableVectorized(const Loop *L, const LoopInfo &LI) {
  for (const BasicBlock *BB : L->getBlocks()) {
    if (LI.getLoopFor(BB) != L) continue;
    for (const Instruction &I : *BB) {
      auto scalable = [](Type *T) {
        auto *VT = dyn_cast<VectorType>(T);
        return VT && VT->getElementCount().isScalable();
      };
      if (const auto *Ld = dyn_cast<LoadInst>(&I))
        if (scalable(Ld->getType())) return true;
      if (const auto *St = dyn_cast<StoreInst>(&I))
        if (scalable(St->getValueOperand()->getType())) return true;
    }
  }
  return false;
}

static bool isAnyVectorized(const Loop *L, const LoopInfo &LI) {
  for (const BasicBlock *BB : L->getBlocks()) {
    if (LI.getLoopFor(BB) != L) continue;
    for (const Instruction &I : *BB) {
      if (const auto *Ld = dyn_cast<LoadInst>(&I))
        if (Ld->getType()->isVectorTy()) return true;
      if (const auto *St = dyn_cast<StoreInst>(&I))
        if (St->getValueOperand()->getType()->isVectorTy()) return true;
    }
  }
  return false;
}

static bool sveGuardFires(const Loop *L, const LoopInfo &LI,
                           ScalarEvolution &SE, SveGuardMode Mode) {
  if (Mode == SveGuardMode::Disabled) return false;
  if (Mode == SveGuardMode::Full)
    return isa<SCEVCouldNotCompute>(SE.getBackedgeTakenCount(L));
  if (!L->isInnermost()) return false;
  if (!isScalableVectorized(L, LI)) return false;
  return isa<SCEVCouldNotCompute>(SE.getBackedgeTakenCount(L));
}

// ==========================================================================
// §3  Strided-access helpers
// ==========================================================================
static int64_t constStride(Value *Ptr, ScalarEvolution &SE, const Loop *L) {
  const auto *AR = dyn_cast<SCEVAddRecExpr>(SE.getSCEV(Ptr));
  if (!AR || !AR->isAffine()) return 0;
  if (const auto *C = dyn_cast<SCEVConstant>(AR->getStepRecurrence(SE)))
    return C->getAPInt().getSExtValue();
  return 0;
}

/// Estimate backedge-taken count as a constant, or 0 if unknown.
static uint64_t knownTripCount(const Loop *L, ScalarEvolution &SE) {
  const SCEV *BTC = SE.getBackedgeTakenCount(L);
  if (isa<SCEVCouldNotCompute>(BTC)) return 0;
  if (const auto *C = dyn_cast<SCEVConstant>(BTC))
    return C->getAPInt().getLimitedValue(UINT64_MAX);
  // Not a constant but computable – use the upper bound of the signed range.
  ConstantRange CR = SE.getSignedRange(BTC);
  if (CR.isFullSet() || CR.isEmptySet()) return 0;
  return CR.getUpper().getLimitedValue(UINT64_MAX);
}

enum class AccessSem { HighReuse, ModReuse, Streaming, Invariant, Unknown };

// Stride-to-semantic mapping (v2 – aggressive):
//
//  Abs == 0           → Unknown  (SCEV failed; already filtered upstream)
//  Invariant          → Invariant (loop-invariant ptr; skip)
//
//  Abs <= 64B (≤ 1 cache line):
//    outer loop → HighReuse  (column reuse across outer iterations)
//    inner loop → Streaming  (HW PF handles dense sequential streams)
//
//  64B < Abs ≤ 512B (moderate stride; HW PF blind):
//    → ModReuse  (SW prefetch into L1 on Oryon, L2 on Ampere)
//    These are the strides that benefit most from SW prefetch.
//    strided_16: 16 doubles × 8B = 128B → falls here correctly.
//
//  Abs > 512B:
//    outer loop → HighReuse  (outer-product / matrix-column reuse)
//    inner loop → ModReuse   (large-stride inner; L2 conservative)
static AccessSem classifyLoad(LoadInst *LI, ScalarEvolution &SE,
                               const Loop *L) {
  Value *Ptr = LI->getPointerOperand();
  if (SE.isLoopInvariant(SE.getSCEV(Ptr), L)) return AccessSem::Invariant;
  int64_t Abs = std::abs(constStride(Ptr, SE, L));
  if (Abs == 0) return AccessSem::Unknown;
  bool outer = (L->getParentLoop() != nullptr);

  if (Abs <= 64 && !outer) return AccessSem::Streaming;
  if (Abs <= 64 &&  outer) return AccessSem::HighReuse;
  if (Abs <= 512)          return AccessSem::ModReuse;
  if (outer)               return AccessSem::HighReuse;
  return                          AccessSem::ModReuse;
}

// Locality: 3=PLDL1KEEP, 2=PLDL2KEEP, 1=PLDL1STRM, -1=skip
//
//  Oryon (10-wide OOO, 4 LD units, 64KB L1, 512KB L2):
//    HighReuse → 3 (L1KEEP): column reuse, want it hot in L1
//    ModReuse  → 3 (L1KEEP): 128B-stride fits L1; L2 caused +22% regression
//    Streaming → 1 (L1STRM): HW already handles; STRM avoids L1 pollution
//
//  Ampere (4-wide, 2 LD units, 64KB L1, 1MB L2, strong HW PF):
//    HighReuse → 2 (L2KEEP): narrower machine; L2 staging is safer
//    ModReuse  → 2 (L2KEEP): conservative
//    Streaming → -1 (skip):  Ampere HW PF handles sequential perfectly
static int32_t locality(AccessSem Sem, const TargetProfile &P) {
  bool oryon = (P.kind == TargetKind::SnapdragonElite);
  switch (Sem) {
  case AccessSem::HighReuse: return oryon ? 3 : 2;
  case AccessSem::ModReuse:  return oryon ? 3 : 2;
  case AccessSem::Streaming: return oryon ? 1 : -1;
  default: return -1;
  }
}

// Adaptive lookahead distance (bytes).
//
// Fixed lookahead (= pf_lookahead_iters × stride) breaks in two ways:
//
//  1. Small stride (8B): lookahead=16 → dist=128B  ✓ reasonable
//     But the same multiplier for stride=128B → dist=2048B, which
//     prefetches 32 cache lines ahead into a 64KB L1.  For a 8MB array
//     (strided_16 benchmark) that's fine, BUT for a short loop the
//     prefetch may land outside the array, wasting bandwidth.
//
//  2. Short loops (trip count < lookahead): prefetching beyond the array
//     end causes TLB misses and cache pollution with no benefit.
//     Guard: if knownTripCount > 0 and dist > TC × stride, clamp dist.
//
// Formula (Oryon, stride S, trip count TC):
//   raw_dist = min(lookahead_iters, TC/2) × S        [TC guard]
//   raw_dist = clamp(raw_dist, min_dist, max_dist)   [profile limits]
//
// The TC/2 divisor keeps prefetch within the second half of the remaining
// iterations – aggressive enough to hide L2 latency (~16 cycles on Oryon)
// without overshooting the working set.
static int64_t adaptiveDist(int64_t AbsStride, uint64_t TC,
                             const TargetProfile &P) {
  uint32_t iters = P.pf_lookahead_iters;
  // Trip-count guard: never prefetch more than TC/2 iterations ahead.
  if (TC > 0 && (uint64_t)iters > TC / 2)
    iters = (uint32_t)std::max((uint64_t)1, TC / 2);
  int64_t raw = (int64_t)iters * AbsStride;
  return std::clamp(raw,
                    (int64_t)P.budget.min_dist_bytes,
                    (int64_t)P.budget.max_dist_bytes);
}

// ==========================================================================
// §4  Gather / SpMV pattern
// ==========================================================================
struct GatherPat {
  LoadInst *idx_ld;
  LoadInst *gather_ld;
  Value    *gather_base;
  int64_t   idx_stride;
};

static bool isExternal(Value *V) {
  V = V->stripPointerCastsAndAliases();
  return isa<GlobalValue>(V) || isa<Argument>(V) || isa<CallBase>(V);
}

static SmallVector<GatherPat, 4>
findGatherPats(Loop *L, ScalarEvolution &SE, const LoopInfo &LI, unsigned Max) {
  SmallVector<GatherPat, 4> Out;
  for (BasicBlock *BB : L->getBlocks()) {
    if (LI.getLoopFor(BB) != L) continue;
    for (Instruction &I : *BB) {
      if (Out.size() >= Max) return Out;
      auto *GLD = dyn_cast<LoadInst>(&I);
      if (!GLD || !GLD->isSimple()) continue;
      auto *GEP = dyn_cast<GetElementPtrInst>(
          GLD->getPointerOperand()->stripPointerCasts());
      if (!GEP || !isExternal(GEP->getPointerOperand())) continue;
      Value *VIdx = nullptr;
      for (auto &U : GEP->indices()) {
        Value *Idx = U.get();
        if (isa<Constant>(Idx) || SE.isLoopInvariant(SE.getSCEV(Idx), L))
          continue;
        if (VIdx) { VIdx = nullptr; break; }
        VIdx = Idx;
      }
      if (!VIdx) continue;
      if (auto *Cast = dyn_cast<CastInst>(VIdx)) VIdx = Cast->getOperand(0);
      auto *IdxLD = dyn_cast<LoadInst>(VIdx);
      if (!IdxLD || !IdxLD->isSimple() || !isExternal(IdxLD->getPointerOperand()))
        continue;
      int64_t IdxStr = constStride(IdxLD->getPointerOperand(), SE, L);
      if (IdxStr == 0) continue;
      Out.push_back({IdxLD, GLD, GEP->getPointerOperand(), IdxStr});
    }
  }
  return Out;
}

// ==========================================================================
// §5  SpMV reorder detection
// ==========================================================================
static SpMVReorderHint reorderHint(const Loop *L, ScalarEvolution &SE) {
  if (MDNode *LID = L->getLoopID()) {
    for (const MDOperand &Op : LID->operands()) {
      auto *T = dyn_cast<MDNode>(Op);
      if (!T || T->getNumOperands() < 2) continue;
      auto *K = dyn_cast<MDString>(T->getOperand(0));
      if (!K || K->getString() != "ahp.spmv.reordered") continue;
      auto *V = dyn_cast<ConstantAsMetadata>(T->getOperand(1));
      if (!V) continue;
      return cast<ConstantInt>(V->getValue())->isOne()
             ? SpMVReorderHint::Reordered : SpMVReorderHint::Random;
    }
  }
  const SCEV *BTC = SE.getBackedgeTakenCount(L);
  if (!isa<SCEVCouldNotCompute>(BTC)) {
    ConstantRange CR = SE.getSignedRange(BTC);
    if (!CR.isFullSet() && !CR.isEmptySet() &&
        (CR.getUpper() - CR.getLower()).ule(64))
      return SpMVReorderHint::Reordered;
  }
  return SpMVReorderHint::Unknown;
}

// ==========================================================================
// §6  Prefetch emission helpers
//
// INSERTION RULE: All prefetches are inserted BEFORE the load instruction
// they service (inline in the loop body). This is always safe:
//   - The pointer operand of the load is live at the load's position.
//   - Adding a GEP + call before the load does not violate dominance.
//   - We apply the lookahead offset via a constant byte GEP.
//
// We intentionally do NOT use the loop preheader to avoid the dominance
// violation that occurs when a vectorized loop's pointer GEPs are inside
// the loop body (they would not dominate a preheader insertion).
// ==========================================================================
static FunctionCallee getPrefetchFn(Module *M) {
  LLVMContext &Ctx = M->getContext();
  return Intrinsic::getOrInsertDeclaration(
      M, Intrinsic::prefetch, {PointerType::getUnqual(Ctx)});
}

/// Insert a single prefetch for Ptr+OffsetBytes before InsertBefore.
static void emitPrefetch(Instruction *InsertBefore, Value *Ptr,
                          int64_t OffsetBytes, uint32_t Locality,
                          FunctionCallee PF) {
  IRBuilder<> B(InsertBefore);
  Type *I8  = Type::getInt8Ty(InsertBefore->getContext());
  Type *I64 = Type::getInt64Ty(InsertBefore->getContext());
  Value *Addr = B.CreateGEP(I8, Ptr, ConstantInt::get(I64, OffsetBytes));
  B.CreateCall(PF, {Addr, B.getInt32(0), B.getInt32(Locality), B.getInt32(1)});
}

// ==========================================================================
// §7  Gather prefetch injection
// ==========================================================================
static unsigned injectGatherPF(const GatherPat &Pat, const TargetProfile &P,
                                Module *M, SpMVReorderHint Hint,
                                uint32_t &Budget) {
  if (!Budget || P.gather_pf_mode == GatherPrefetchMode::Disabled) return 0;
  FunctionCallee PF = getPrefetchFn(M);
  unsigned N = 0;
  bool agg = (Hint == SpMVReorderHint::Reordered);

  // A: prefetch col_idx stream ahead (insert before idx_ld)
  if (Budget > 0) {
    int64_t Dist = std::clamp(
        (int64_t)P.gather_lookahead * std::abs(Pat.idx_stride),
        (int64_t)P.budget.min_dist_bytes, (int64_t)P.budget.max_dist_bytes);
    emitPrefetch(Pat.idx_ld, Pat.idx_ld->getPointerOperand(), Dist,
                 1 /*PLDL1STRM*/, PF);
    --Budget; ++N;
  }

  // B: prefetch x[col_idx[j]] current (insert before gather_ld)
  if (Budget > 0) {
    emitPrefetch(Pat.gather_ld, Pat.gather_ld->getPointerOperand(),
                 0 /*current addr, no offset*/, P.gather_locality, PF);
    --Budget; ++N;
  }

  // C: spatial +64B for reordered matrices (insert before gather_ld)
  if (agg && Budget > 0) {
    emitPrefetch(Pat.gather_ld, Pat.gather_ld->getPointerOperand(),
                 64, P.gather_locality, PF);
    --Budget; ++N;
  }

  return N;
}

// ==========================================================================
// §8  AArch64LaneHintPass (LoopPass @ LoopOptimizerEndEP)
// ==========================================================================
static bool hasFPReduction(const Loop *L) {
  for (PHINode &Phi : L->getHeader()->phis()) {
    if (!Phi.getType()->isFloatingPointTy()) continue;
    for (unsigned i = 0; i < Phi.getNumIncomingValues(); i++) {
      if (!L->contains(Phi.getIncomingBlock(i))) continue;
      Value *V = Phi.getIncomingValue(i);
      if (!isa<Instruction>(V)) continue;
      unsigned Op = cast<Instruction>(V)->getOpcode();
      if (Op == Instruction::FAdd || Op == Instruction::FSub ||
          Op == Instruction::FMul || Op == Instruction::Call)
        return true;
    }
  }
  return false;
}

static void applyLaneHintMD(Loop *L, const IndependentLanePolicy &Pol,
                             LLVMContext &Ctx) {
  if (!Pol.inject_unroll_hint && !Pol.inject_interleave) return;
  MDNode *Old = L->getLoopID();
  SmallVector<Metadata *, 8> MDs;
  MDs.push_back(nullptr);
  if (Old) {
    for (unsigned i = 1, e = Old->getNumOperands(); i < e; ++i) {
      auto *Op = dyn_cast<MDNode>(Old->getOperand(i));
      if (!Op) continue;
      auto *KS = dyn_cast<MDString>(Op->getOperand(0));
      if (!KS) { MDs.push_back(Op); continue; }
      StringRef K = KS->getString();
      if (K == "llvm.loop.unroll.count" || K == "llvm.loop.interleave.count")
        continue;
      MDs.push_back(Op);
    }
  }
  Type *I32 = Type::getInt32Ty(Ctx);
  const uint8_t UF = Pol.unroll_factor;
  auto addMD = [&](StringRef Key, uint8_t Val) {
    Metadata *Ops[] = {MDString::get(Ctx, Key),
                       ConstantAsMetadata::get(ConstantInt::get(I32, Val))};
    MDs.push_back(MDNode::get(Ctx, Ops));
  };
  if (Pol.inject_unroll_hint) addMD("llvm.loop.unroll.count",    UF);
  if (Pol.inject_interleave)  addMD("llvm.loop.interleave.count", UF);
  MDNode *New = MDNode::get(Ctx, MDs);
  New->replaceOperandWith(0, New);
  L->setLoopID(New);
}

class AArch64LaneHintPass : public PassInfoMixin<AArch64LaneHintPass> {
public:
  PreservedAnalyses run(Loop &L, LoopAnalysisManager &,
                        LoopStandardAnalysisResults &, LPMUpdater &) {
    Function *F = L.getHeader()->getParent();
    if (!F || F->isDeclaration()) return PreservedAnalyses::all();
    const TargetProfile P = detectTarget(*F);
    if (!P.lane_policy.enabled || !L.isInnermost() || !hasFPReduction(&L))
      return PreservedAnalyses::all();
    applyLaneHintMD(&L, P.lane_policy, F->getContext());
    ++NumLaneHintsInjected;
    LLVM_DEBUG(dbgs() << "[LaneHint] UF=" << (int)P.lane_policy.unroll_factor
                      << " on " << F->getName() << "\n");
    PreservedAnalyses PA;
    PA.preserveSet<CFGAnalyses>();
    return PA;
  }
};

// ==========================================================================
// §9  AArch64HeteroOptPass (FunctionPass @ OptimizerLastEP)
// ==========================================================================
PreservedAnalyses
AArch64HeteroOptPass::run(Function &F, FunctionAnalysisManager &FAM) {
  if (F.isDeclaration() || F.isIntrinsic()) return PreservedAnalyses::all();

  auto &LI = FAM.getResult<LoopAnalysis>(F);
  auto &SE = FAM.getResult<ScalarEvolutionAnalysis>(F);
  const TargetProfile P = detectTarget(F);

  LLVM_DEBUG(dbgs() << "[HeteroOpt] " << F.getName()
                    << " cpu=" << P.cpu_name
                    << " sve=" << (int)P.sve_guard
                    << " gather=" << (int)P.gather_pf_mode << "\n");

  if (!P.inject_sw_prefetch) return PreservedAnalyses::all();

  bool     Changed  = false;
  uint32_t FnBudget = P.budget.per_function;
  Module  *M        = F.getParent();
  FunctionCallee PF = getPrefetchFn(M);

  for (Loop *L : LI.getLoopsInPreorder()) {
    if (FnBudget == 0) { ++NumLoopsSkippedBudget; break; }

    if (sveGuardFires(L, LI, SE, P.sve_guard)) {
      ++NumLoopsSkippedSVE; continue;
    }

    bool LoopChanged = false;

    // ── Gather / SpMV prefetch ─────────────────────────────────────────
    if (P.gather_pf_mode != GatherPrefetchMode::Disabled && L->isInnermost()) {
      auto Pats = findGatherPats(L, SE, LI, P.budget.per_loop_gather);
      if (!Pats.empty()) {
        ++NumGatherLoopsDetected;
        SpMVReorderHint Hint = reorderHint(L, SE);
        uint32_t GB = std::min(FnBudget, P.budget.per_loop_gather);
        for (const auto &Pat : Pats) {
          if (!GB) break;
          unsigned n = injectGatherPF(Pat, P, M, Hint, GB);
          if (n) {
            NumGatherPFInjected += n;
            FnBudget = (FnBudget > n) ? FnBudget - n : 0;
            LoopChanged = true;
          }
        }
      }
    }

    // ── Strided prefetch (skip vectorized loops) ───────────────────────
    if (isAnyVectorized(L, LI)) {
      ++NumLoopsSkippedVec;
      if (LoopChanged) { ++NumLoopsOptimized; Changed = true; }
      continue;
    }

    uint32_t LpBudget = P.budget.per_loop;

    for (BasicBlock *BB : L->getBlocks()) {
      if (LI.getLoopFor(BB) != L) continue;
      if (!FnBudget || !LpBudget) break;

      for (Instruction &I : *BB) {
        if (!FnBudget || !LpBudget) break;
        auto *LD = dyn_cast<LoadInst>(&I);
        if (!LD || !LD->isSimple()) continue;

        AccessSem Sem = classifyLoad(LD, SE, L);
        if (Sem == AccessSem::Invariant) { ++NumLoadsSkippedInvariant; continue; }
        if (Sem == AccessSem::Unknown)   continue;
        int32_t Loc = locality(Sem, P);
        if (Loc < 0) { ++NumLoadsSkippedStreaming; continue; }

        int64_t Stride = constStride(LD->getPointerOperand(), SE, L);
        if (!Stride) continue;

        // Compute adaptive prefetch distance.
        // knownTripCount = 0 means "unknown / unbounded" → use full lookahead.
        uint64_t TC = knownTripCount(L, SE);
        int64_t Dist = adaptiveDist(std::abs(Stride), TC, P);

        // Random-access guard: if the trip count is known and the total
        // working set (TC × |stride|) fits in L1, the HW prefetcher and
        // OOO window will handle it.  SW prefetch adds overhead for no gain.
        //
        // Threshold: Oryon L1D = 64 KB.  If TC × |stride| < L1D_size/2,
        // the entire working set fits comfortably and SW prefetch hurts.
        // This is the source of the random_access +6% regression:
        // arr[idx[i]] has loop-variant GEP but the outer *idx* traversal
        // is sequential with stride=8B.  SCEV sees it as Streaming (Abs=8,
        // !outer → skip) but on some compilation paths the inner GEP's
        // pointer gets a different SCEV expression and slips through with
        // Loc≥0.  The working-set guard catches it unconditionally.
        {
          uint32_t L1KB = P.primary.cache.l1d_size_kb;
          int64_t WS = (TC > 0) ? (int64_t)TC * std::abs(Stride) : INT64_MAX;
          if (WS < (int64_t)(L1KB * 1024 / 2)) {
            ++NumLoadsSkippedInvariant; // reuse existing counter (fits in cache)
            continue;
          }
        }

        // Insert BEFORE the load (inline in loop body, always dominates).
        emitPrefetch(&I, LD->getPointerOperand(), Dist, (uint32_t)Loc, PF);

        --FnBudget; --LpBudget; ++NumStridedPFInjected;
        LoopChanged = true;
        LLVM_DEBUG(dbgs() << "[HeteroOpt] strided pf loc=" << Loc
                          << " dist=" << Dist << "B stride=" << Stride
                          << " TC=" << TC << "\n");
      }
    }

    if (LoopChanged) { ++NumLoopsOptimized; Changed = true; }
  }

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

// ==========================================================================
// §10  Plugin registration
// ==========================================================================
extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {
    LLVM_PLUGIN_API_VERSION, "AArch64HeteroOpt", LLVM_VERSION_STRING,
    [](::llvm::PassBuilder &PB) {
      PB.registerPipelineParsingCallback(
        [](StringRef Name, FunctionPassManager &FPM,
           ArrayRef<PassBuilder::PipelineElement>) -> bool {
          if (Name == "aarch64-hetero-opt") {
            FPM.addPass(AArch64HeteroOptPass()); return true;
          }
          return false;
        });
      // LaneHintPass at LoopOptimizerEndEP: pre-vectorizer, safe for setLoopID.
      PB.registerLoopOptimizerEndEPCallback(
        [](LoopPassManager &LPM, OptimizationLevel Opt) {
          if (Opt != OptimizationLevel::O0)
            LPM.addPass(AArch64LaneHintPass());
        });
      // HeteroOptPass at OptimizerLastEP: post-vectorizer, prefetch intrinsics only.
      PB.registerOptimizerLastEPCallback(
        [](ModulePassManager &MPM, OptimizationLevel Opt, ThinOrFullLTOPhase) {
          if (Opt == OptimizationLevel::O0) return;
          FunctionPassManager FPM;
          FPM.addPass(AArch64HeteroOptPass());
          MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
        });
    }
  };
}

#endif // HAS_LLVM_HEADERS
