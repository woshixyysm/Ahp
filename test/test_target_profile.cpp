//===- test_target_profile.cpp - Tests for AArch64TargetProfile.h ---------===//
//
// Validates target profile fields, locality decision table, prefetch budget,
// SVE guard mode, gather prefetch policy, and independent-lane policy.
// No LLVM dependency required.
//
//===----------------------------------------------------------------------===//

#include "AArch64TargetProfile.h"
#include <cassert>
#include <cstdint>
#include <cstdio>

using namespace aarch64hetero;

static int failures = 0;

#define CHECK(expr, msg)                                                        \
  do {                                                                          \
    if (!(expr)) {                                                               \
      fprintf(stderr, "[FAIL] %s  (%s:%d)\n", msg, __FILE__, __LINE__);         \
      ++failures;                                                                \
    } else {                                                                     \
      printf("[PASS] %s\n", msg);                                                \
    }                                                                            \
  } while (0)

// ---------------------------------------------------------------------------
// Locality table (mirrors AArch64HeteroOpt.cpp)
// ---------------------------------------------------------------------------
enum class AccessSemantic { HighReuse, ModerateReuse, Streaming, Invariant, Unknown };

static int32_t localityForAccess(AccessSemantic Sem, const TargetProfile &P) {
  const bool isOryon = (P.kind == TargetKind::SnapdragonElite);
  switch (Sem) {
  case AccessSemantic::HighReuse:     return isOryon ? 3 : 2;
  case AccessSemantic::ModerateReuse: return 2;
  case AccessSemantic::Streaming:     return isOryon ? 1 : -1;
  case AccessSemantic::Invariant:
  case AccessSemantic::Unknown:       return -1;
  }
  return -1;
}

// ---------------------------------------------------------------------------
// Profile field tests
// ---------------------------------------------------------------------------
static void test_ampere_profile() {
  TargetProfile p = makeAmpereAltraProfile();
  CHECK(p.kind                           == TargetKind::AmpereAltra, "Ampere kind");
  CHECK(p.cpu_name                       == "ampere1",               "Ampere cpu_name");
  CHECK(p.primary.cache.l1d_size_kb      == 64u,                     "Ampere L1D=64KB");
  CHECK(p.primary.cache.l2_size_kb       == 1024u,                   "Ampere L2=1MB");
  CHECK(p.primary.cache.l3_size_mb       == 32u,                     "Ampere L3=32MB");
  CHECK(p.primary.pipeline.issue_width   == 4u,                      "Ampere IW=4");
  CHECK(p.primary.pipeline.has_sme       == false,                   "Ampere no SME");
  CHECK(p.primary.pipeline.has_sve2      == false,                   "Ampere no SVE2");
  CHECK(p.inject_sw_prefetch             == false,                   "Ampere no SW PF");
  CHECK(p.pf_lookahead_iters             == 24u,                     "Ampere lookahead=24");
  CHECK(p.use_prfm_pldl2keep             == true,                    "Ampere pldl2keep");
  CHECK(!p.efficiency.has_value(),                                    "Ampere homogeneous");
  CHECK(p.budget.per_function            == 16u,                     "Ampere budget fn=16");
  CHECK(p.budget.per_loop                == 4u,                      "Ampere budget loop=4");
  CHECK(p.budget.max_dist_bytes          == 4096u,                   "Ampere max_dist=4096");
  // SVE guard: Disabled (no SVE2 on Ampere, no tail-fold risk)
  CHECK(p.sve_guard == SveGuardMode::Disabled,                       "Ampere SVE guard Disabled");
  // Gather: disabled on Ampere (HW prefetcher handles streaming)
  CHECK(p.gather_pf_mode == GatherPrefetchMode::Disabled,            "Ampere gather disabled");
  // Lane policy: disabled on Ampere
  CHECK(p.lane_policy.enabled == false,                              "Ampere lane policy off");
}

static void test_oryon_profile() {
  TargetProfile p = makeSnapdragon8EliteProfile();
  CHECK(p.kind                           == TargetKind::SnapdragonElite, "Oryon kind");
  CHECK(p.cpu_name                       == "oryon-1",                   "Oryon cpu_name");
  CHECK(p.primary.cache.l1d_size_kb      == 64u,                        "Oryon P L1D=64KB");
  CHECK(p.primary.cache.l2_size_kb       == 512u,                       "Oryon P L2=512KB");
  CHECK(p.primary.cache.l3_size_mb       == 12u,                        "Oryon L3=12MB");
  CHECK(p.primary.pipeline.issue_width   == 10u,                        "Oryon P IW=10");
  CHECK(p.primary.pipeline.has_sme       == true,                       "Oryon P has SME");
  CHECK(p.primary.pipeline.has_sve2      == true,                       "Oryon P has SVE2");
  CHECK(p.inject_sw_prefetch             == true,                       "Oryon SW PF on");
  CHECK(p.pf_lookahead_iters             == 8u,                         "Oryon lookahead=8");
  CHECK(p.use_prfm_pldl1keep             == true,                       "Oryon pldl1keep");
  CHECK(p.efficiency.has_value(),                                         "Oryon has E-core");
  CHECK(p.efficiency->cache.l2_size_kb   == 256u,                       "Oryon E L2=256KB");
  CHECK(p.efficiency->pipeline.has_sme   == false,                      "Oryon E no SME");
  CHECK(p.budget.per_function            == 32u,                        "Oryon budget fn=32");
  CHECK(p.budget.per_loop                == 8u,                         "Oryon budget loop=8");
  CHECK(p.budget.per_loop_gather         == 4u,                         "Oryon gather budget=4");
  CHECK(p.budget.max_dist_bytes          == 8192u,                      "Oryon max_dist=8192");
  // SVE guard: InnerOnly — SpMV fix
  CHECK(p.sve_guard == SveGuardMode::InnerOnly,                         "Oryon SVE guard InnerOnly");
  // Gather: enabled (NontemporalHint at minimum)
  CHECK(p.gather_pf_mode != GatherPrefetchMode::Disabled,               "Oryon gather enabled");
  CHECK(p.gather_locality == 3u,                                        "Oryon gather loc=3 (L1)");
  CHECK(p.gather_lookahead == 4u,                                       "Oryon gather lookahead=4");
  // Lane policy: enabled for accumulator unrolling
  CHECK(p.lane_policy.enabled == true,                                  "Oryon lane policy on");
  CHECK(p.lane_policy.unroll_factor == 4u,                              "Oryon unroll_factor=4");
  CHECK(p.lane_policy.inject_unroll_hint == true,                       "Oryon unroll hint on");
  CHECK(p.lane_policy.inject_interleave == true,                        "Oryon interleave on");
}

// ---------------------------------------------------------------------------
// SVE guard mode semantics
// ---------------------------------------------------------------------------
static void test_sve_guard_semantics() {
  // The critical fix: InnerOnly must NOT skip scalar loops with dynamic TC.
  // We can't invoke real SCEV here, but we can verify the enum values and
  // that Ampere stays Disabled while Oryon uses InnerOnly.
  auto ampere = makeAmpereAltraProfile();
  auto oryon  = makeSnapdragon8EliteProfile();

  CHECK(ampere.sve_guard == SveGuardMode::Disabled,
        "Ampere: Disabled (no SVE2, no risk)");
  CHECK(oryon.sve_guard  == SveGuardMode::InnerOnly,
        "Oryon: InnerOnly (fix for SpMV 34% regression)");

  // Full guard is reserved for when LLVM upstream fixes the bug.
  // Neither target should use it by default.
  CHECK(ampere.sve_guard != SveGuardMode::Full, "Ampere not Full");
  CHECK(oryon.sve_guard  != SveGuardMode::Full, "Oryon not Full");
}

// ---------------------------------------------------------------------------
// Locality decision table
// ---------------------------------------------------------------------------
static void test_locality_table() {
  TargetProfile oryon  = makeSnapdragon8EliteProfile();
  TargetProfile ampere = makeAmpereAltraProfile();

  CHECK(localityForAccess(AccessSemantic::HighReuse, oryon)  == 3, "HighReuse Oryon→3");
  CHECK(localityForAccess(AccessSemantic::HighReuse, ampere) == 2, "HighReuse Ampere→2");
  CHECK(localityForAccess(AccessSemantic::ModerateReuse, oryon)  == 2, "ModReuse Oryon→2");
  CHECK(localityForAccess(AccessSemantic::ModerateReuse, ampere) == 2, "ModReuse Ampere→2");
  CHECK(localityForAccess(AccessSemantic::Streaming, oryon)  == 1, "Streaming Oryon→1");
  CHECK(localityForAccess(AccessSemantic::Streaming, ampere) == -1, "Streaming Ampere→skip");
  CHECK(localityForAccess(AccessSemantic::Invariant, oryon)  == -1, "Invariant Oryon→skip");
  CHECK(localityForAccess(AccessSemantic::Invariant, ampere) == -1, "Invariant Ampere→skip");
  CHECK(localityForAccess(AccessSemantic::Unknown, oryon)    == -1, "Unknown Oryon→skip");
  CHECK(localityForAccess(AccessSemantic::Unknown, ampere)   == -1, "Unknown Ampere→skip");
}

// ---------------------------------------------------------------------------
// Budget and clamp
// ---------------------------------------------------------------------------
static void test_budget() {
  auto ampere = makeAmpereAltraProfile();
  auto oryon  = makeSnapdragon8EliteProfile();

  CHECK(ampere.budget.max_dist_bytes < oryon.budget.max_dist_bytes,
        "Ampere max_dist < Oryon max_dist");
  CHECK(ampere.budget.min_dist_bytes == oryon.budget.min_dist_bytes,
        "Same min_dist floor");
  // Oryon got larger budgets to support SpMV gather
  CHECK(oryon.budget.per_function > ampere.budget.per_function,
        "Oryon fn budget > Ampere");
  CHECK(oryon.budget.per_loop_gather > 0u,
        "Oryon gather budget > 0");
}

// ---------------------------------------------------------------------------
// Independent-lane policy
// ---------------------------------------------------------------------------
static void test_lane_policy() {
  auto oryon = makeSnapdragon8EliteProfile();
  // Unroll factor must match load_units (4) to hide FMA latency chain.
  CHECK(oryon.lane_policy.unroll_factor ==
        oryon.primary.pipeline.load_units,
        "Oryon UF == load_units (saturates FP units)");
  CHECK(oryon.lane_policy.min_trip_count >= 4u,
        "min_trip_count ≥ 4 (avoid bloat on tiny loops)");
}

// ---------------------------------------------------------------------------
// Cache arithmetic
// ---------------------------------------------------------------------------
static void test_cache_arithmetic() {
  auto p = makeAmpereAltraProfile();
  CHECK((uint64_t)p.primary.cache.l1d_size_kb * 1024 == 65536u,   "L1D=64KiB");
  CHECK((uint64_t)p.primary.cache.l2_size_kb  * 1024 == 1048576u, "L2=1MiB");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
  printf("=== AArch64TargetProfile + policy tests ===\n\n");

  printf("-- Profile field correctness --\n");
  test_ampere_profile();
  test_oryon_profile();
  test_cache_arithmetic();

  printf("\n-- SVE guard semantics --\n");
  test_sve_guard_semantics();

  printf("\n-- Locality decision table --\n");
  test_locality_table();

  printf("\n-- Budget and clamp --\n");
  test_budget();

  printf("\n-- Independent-lane policy --\n");
  test_lane_policy();

  printf("\n%s\n", failures == 0 ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
  return failures == 0 ? 0 : 1;
}
