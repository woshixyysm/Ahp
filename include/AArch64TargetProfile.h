#pragma once

#include <cstdint>
#include <optional>
#include <string_view>

/// Ahp – AArch64 Hetero-Opt Pass
/// Target-profile data structures and factory functions (pure C++, no LLVM).

namespace aarch64hetero {

// ---------------------------------------------------------------------------
// Logical target class
// ---------------------------------------------------------------------------
enum class TargetKind : uint8_t {
  Unknown,
  AmpereAltra,
  SnapdragonElite,
};

// ---------------------------------------------------------------------------
// Cache / pipeline parameters
// ---------------------------------------------------------------------------
struct CacheParameters {
  uint32_t l1d_size_kb;
  uint32_t l1d_line_bytes;
  uint32_t l2_size_kb;
  uint32_t l3_size_mb;
  uint32_t prefetch_distance;
  uint32_t prefetch_stride;
  bool     has_hardware_pf;
  bool     prefers_nontemporal;
  uint32_t nt_store_threshold_kb;
};

struct PipelineParameters {
  uint8_t issue_width;
  uint8_t load_units;
  uint8_t store_units;
  uint8_t branch_penalty;
  uint8_t fp_latency_fma;
  uint8_t mem_latency_l1;
  uint8_t mem_latency_l2;
  bool    out_of_order;
  bool    has_sme;
  bool    has_sve2;
  bool    has_mops;
};

struct CoreCluster {
  std::string_view   name;
  uint8_t            core_count;
  CacheParameters    cache;
  PipelineParameters pipeline;
};

// ---------------------------------------------------------------------------
// SW-prefetch injection budget
// ---------------------------------------------------------------------------
struct PrefetchBudget {
  uint32_t per_function;
  uint32_t per_loop;
  uint32_t min_dist_bytes;
  uint32_t max_dist_bytes;
  /// Additional budget slots reserved for gather / indirect-access prefetch.
  /// These are counted separately from strided prefetches so that a function
  /// with both sparse and dense loops is not starved on one side.
  uint32_t per_loop_gather;
};

// ---------------------------------------------------------------------------
// Gather (indirect / sparse) prefetch policy
//
// SpMV / BFS / graph algorithms share a pattern:
//   for j in row_ptr[i]..row_ptr[i+1]:        ← inner loop, dynamic TC
//     y[i] += val[j] * x[col_idx[j]]          ← gather from x[]
//
// Two prefetch strategies for the gather target x[col_idx[j]]:
//
//   SoftwareGatherPrefetch (SGP):
//     Insert prfm pldl1keep for x[col_idx[j+D]] where D = gather_lookahead.
//     Requires: compile-time-visible gather index (col_idx must be a load
//     whose address is an affine AddRec).
//     Works when: col_idx[] is sequential (reordered / BFS-ordered matrix).
//     Does NOT work: random col_idx[] (prefetch still fires but hurts).
//
//   NontemporalGatherHint (NGH):
//     Use prfm pldl1strm for val[j] (the streaming operand) and
//     prfm pldl1keep for x[col_idx[j+D]] (gather target).
//     Fires unconditionally – safe even for random col_idx because the
//     gather target prefetch is speculative and just warms TLB + L2.
//
// The pass selects SGP when the matrix is detected as reordered
// (see SpMVReorderHint below) and falls back to NGH otherwise.
// ---------------------------------------------------------------------------
enum class GatherPrefetchMode : uint8_t {
  Disabled,             ///< No gather prefetch (conservative default)
  NontemporalHint,      ///< NGH: prfm pldl1strm val + prfm pldl1keep x[j+D]
  SoftwareGatherPrefetch, ///< SGP: prfm pldl1keep x[col_idx[j+D]]
};

// ---------------------------------------------------------------------------
// SpMV matrix reorder detection hint
//
// The pass detects "reordered matrix" structural patterns from metadata or
// loop structure heuristics:
//   - Loop trip count distribution: constant or small-range BTC → reordered
//   - Row-length variance: tight SCEV range on (row_ptr[i+1]-row_ptr[i])
//   - User annotation: llvm.loop metadata "ahp.spmv.reordered"=true
//
// When reordered, col_idx[j] values fall in a small range → x[] accesses
// are spatially local → aggressive L1 prefetch is safe and profitable.
// ---------------------------------------------------------------------------
enum class SpMVReorderHint : uint8_t {
  Unknown,    ///< No information; use NGH conservatively
  Reordered,  ///< Matrix is reordered (RCM / BFS / AMD); use SGP
  Random,     ///< Explicitly random; disable gather prefetch
};

// ---------------------------------------------------------------------------
// SVE guard mode
//
// Replaces the old binary guard_sve_tail_fold flag with a richer policy:
//
//   Full:        Skip ALL loops with dynamic TC on SVE2 targets.
//                Original behavior – 34% SpMV regression.
//
//   InnerOnly:   Skip SVE vectorized inner loops with dynamic TC.
//                Outer loops and scalar inner loops are NOT skipped.
//                This is the correct fix: SpMV's inner loop is SCALAR
//                after the vectorizer gives up on it (ragged row lengths
//                prevent vectorization). So InnerOnly = safe + profitable.
//
//   Disabled:    No guard (use only when LLVM upstream tail-fold bug is fixed).
// ---------------------------------------------------------------------------
enum class SveGuardMode : uint8_t {
  Full,       ///< Skip all dynamic-TC loops (safe but over-conservative)
  InnerOnly,  ///< Skip only SVE-vectorized inner loops with dynamic TC
  Disabled,   ///< No guard
};

// ---------------------------------------------------------------------------
// Independent-lane unrolling policy
//
// User measurement: dependent-chain time 3.97s vs independent-lanes 1.96s
// → 2.03x speedup potential from eliminating false RAW hazards.
//
// SpMV inner loop accumulates into y[i] through a sequential reduction:
//   y[i] += val[j] * x[col_idx[j]]   ← j=0,1,2,...
//
// Every iteration depends on the previous FMA result (RAW on y[i]).
// On Oryon (10-wide OOO, 4 FP load units), the bottleneck is the FMA
// dependence chain (~4 cycles latency), not memory bandwidth.
//
// Fix: partial unroll with independent accumulators:
//   acc0 += val[j]   * x[col_idx[j]]
//   acc1 += val[j+1] * x[col_idx[j+1]]
//   acc2 += val[j+2] * x[col_idx[j+2]]
//   acc3 += val[j+3] * x[col_idx[j+3]]
//   y[i] = acc0 + acc1 + acc2 + acc3
//
// This hides 3 FMA latencies and saturates the 4 FP load units.
// The pass implements this via loop unroll metadata injection.
// ---------------------------------------------------------------------------
struct IndependentLanePolicy {
  bool    enabled;
  uint8_t unroll_factor;     ///< Partial unroll count (4 = 4 accumulators)
  uint8_t min_trip_count;    ///< Don't unroll if BTC < this (avoids bloat)
  bool    inject_unroll_hint; ///< Inject llvm.loop.unroll.count metadata
  bool    inject_interleave;  ///< Inject llvm.loop.interleave.count metadata
};

// ---------------------------------------------------------------------------
// Full target profile
// ---------------------------------------------------------------------------
struct TargetProfile {
  TargetKind       kind;
  std::string_view cpu_name;
  std::string_view march_str;

  CoreCluster               primary;
  std::optional<CoreCluster> efficiency;

  // Strided SW prefetch
  bool         inject_sw_prefetch;
  uint32_t     pf_lookahead_iters;
  bool         use_prfm_pldl1keep;
  bool         use_prfm_pldl2keep;
  bool         use_prfm_pstl1keep;
  PrefetchBudget budget;

  // SVE tail-folding guard (replaces old bool guard_sve_tail_fold)
  SveGuardMode sve_guard;

  // Gather / sparse prefetch
  GatherPrefetchMode gather_pf_mode;
  uint32_t           gather_lookahead;   ///< How many rows ahead to prefetch
  uint32_t           gather_locality;    ///< llvm.prefetch locality for gather

  // Independent-lane (accumulator unrolling) policy
  IndependentLanePolicy lane_policy;
};

// ---------------------------------------------------------------------------
// Factory: Ampere Altra / AmpereOne
// ---------------------------------------------------------------------------
[[nodiscard]]
inline TargetProfile makeAmpereAltraProfile() noexcept {
  CacheParameters c{
    .l1d_size_kb            = 64,
    .l1d_line_bytes         = 64,
    .l2_size_kb             = 1024,
    .l3_size_mb             = 32,
    .prefetch_distance      = 20,
    .prefetch_stride        = 256,
    .has_hardware_pf        = true,
    .prefers_nontemporal    = true,
    .nt_store_threshold_kb  = 2048,
  };
  PipelineParameters p{
    .issue_width    = 4,
    .load_units     = 2,
    .store_units    = 1,
    .branch_penalty = 11,
    .fp_latency_fma = 4,
    .mem_latency_l1 = 4,
    .mem_latency_l2 = 12,
    .out_of_order   = true,
    .has_sme        = false,
    .has_sve2       = false,
    .has_mops       = false,
  };
  return TargetProfile{
    .kind      = TargetKind::AmpereAltra,
    .cpu_name  = "ampere1",
    .march_str = "armv8.6-a+sve+crypto",
    .primary   = CoreCluster{"Altra", 80, c, p},
    .efficiency = std::nullopt,
    .inject_sw_prefetch  = false,
    .pf_lookahead_iters  = 24,
    .use_prfm_pldl1keep  = false,
    .use_prfm_pldl2keep  = true,
    .use_prfm_pstl1keep  = false,
    .budget = PrefetchBudget{
      .per_function    = 16,
      .per_loop        = 4,
      .min_dist_bytes  = 64,
      .max_dist_bytes  = 4096,
      .per_loop_gather = 2,
    },
    .sve_guard        = SveGuardMode::Disabled, // no SVE2, no risk
    .gather_pf_mode   = GatherPrefetchMode::Disabled,
    .gather_lookahead = 4,
    .gather_locality  = 2,
    .lane_policy = IndependentLanePolicy{
      .enabled             = false,
      .unroll_factor       = 4,
      .min_trip_count      = 8,
      .inject_unroll_hint  = false,
      .inject_interleave   = false,
    },
  };
}

// ---------------------------------------------------------------------------
// Factory: Snapdragon 8 Elite (Oryon P+E)
// ---------------------------------------------------------------------------
[[nodiscard]]
inline TargetProfile makeSnapdragon8EliteProfile() noexcept {
  CacheParameters pcache{
    .l1d_size_kb            = 64,
    .l1d_line_bytes         = 64,
    .l2_size_kb             = 512,
    .l3_size_mb             = 12,
    .prefetch_distance      = 8,
    .prefetch_stride        = 128,
    .has_hardware_pf        = true,
    .prefers_nontemporal    = true,
    .nt_store_threshold_kb  = 512,
  };
  PipelineParameters ppipe{
    .issue_width    = 10,
    .load_units     = 4,
    .store_units    = 2,
    .branch_penalty = 14,
    .fp_latency_fma = 4,
    .mem_latency_l1 = 4,
    .mem_latency_l2 = 16,
    .out_of_order   = true,
    .has_sme        = true,
    .has_sve2       = true,
    .has_mops       = true,
  };
  CacheParameters ecache{
    .l1d_size_kb            = 32,
    .l1d_line_bytes         = 64,
    .l2_size_kb             = 256,
    .l3_size_mb             = 12,
    .prefetch_distance      = 6,
    .prefetch_stride        = 64,
    .has_hardware_pf        = true,
    .prefers_nontemporal    = false,
    .nt_store_threshold_kb  = 1024,
  };
  PipelineParameters epipe{
    .issue_width    = 4,
    .load_units     = 2,
    .store_units    = 1,
    .branch_penalty = 10,
    .fp_latency_fma = 4,
    .mem_latency_l1 = 4,
    .mem_latency_l2 = 12,
    .out_of_order   = true,
    .has_sme        = false,
    .has_sve2       = true,
    .has_mops       = true,
  };
  return TargetProfile{
    .kind      = TargetKind::SnapdragonElite,
    .cpu_name  = "oryon-1",
    .march_str = "armv9.2-a+sve2+sme+mops+crypto",
    .primary   = CoreCluster{"Oryon-P", 4, pcache, ppipe},
    .efficiency = CoreCluster{"Oryon-E", 4, ecache, epipe},
    .inject_sw_prefetch  = true,
    // pf_lookahead_iters = 12 (adaptive):
    //   adaptiveDist() clamps to min(iters, TC/2) × stride, so this is
    //   an upper bound.  Oryon L2 latency ≈16 cycles; 12 iterations of
    //   an 8B-stride loop = 96B (safe, within same page).  For 128B-stride
    //   (strided_16), adaptiveDist clamps to TC/2 iterations ahead, which
    //   is exactly enough to hide L2 latency without overshooting L1.
    .pf_lookahead_iters  = 8,
    .use_prfm_pldl1keep  = true,
    .use_prfm_pldl2keep  = true,
    .use_prfm_pstl1keep  = true,
    .budget = PrefetchBudget{
      .per_function    = 32,
      .per_loop        = 8,
      // min_dist_bytes = 128 (2 cache lines):
      //   The original 64B (= 1 line) caused the same cache line to be
      //   re-prefetched for small-element strides, wasting prefetch bandwidth
      //   and polluting the prefetch queue with no-op requests.
      //   2 lines = 128B ensures every emitted prefetch targets a new line.
      .min_dist_bytes  = 128,
      .max_dist_bytes  = 8192,
      .per_loop_gather = 4,   // generous: SpMV gather is the hot path
    },
    // InnerOnly: SpMV's inner loop is scalar (ragged rows prevent
    // vectorization), so it is NOT skipped. Only SVE-vectorized inner
    // loops with dynamic TC are skipped to avoid the tail-fold livelock.
    .sve_guard = SveGuardMode::InnerOnly,

    // Gather prefetch: NontemporalHint fires unconditionally (safe for
    // any col_idx distribution). Upgraded to SGP when reorder is detected.
    .gather_pf_mode   = GatherPrefetchMode::NontemporalHint,
    .gather_lookahead = 4,    // 4 iterations ahead for gather target
    .gather_locality  = 3,   // PLDL1KEEP: pull gather target into L1

    // Independent-lane policy: unroll factor 4 matches Oryon's 4 FP load
    // units and hides the 4-cycle FMA latency chain (user data: 2.03x).
    .lane_policy = IndependentLanePolicy{
      .enabled             = true,
      .unroll_factor       = 4,
      .min_trip_count      = 8,
      .inject_unroll_hint  = true,
      .inject_interleave   = true,
    },
  };
}

} // namespace aarch64hetero
