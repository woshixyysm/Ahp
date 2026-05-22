# Ahp — AArch64 Hetero-Opt LLVM Pass

针对 LLVM AArch64/Arm64 CPU 的生产级异构优化 Pass。

感谢 ARM、Ampere、Qualcomm、Ming6293、OLLVM、HomuHomu833、IntelICX 提供的优化方向。

---

## Ahp 是什么？

Ahp 是一个 LLVM New Pass Manager 插件，针对以下 AArch64 微架构注入软件预取（SW prefetch）：

| 目标 | CPU 字符串 | 特性 |
|------|------------|------|
| Ampere Altra / AmpereOne | `ampere1`, `ampere1a` | ARMv8.6 + SVE，HW 预取强，禁用 SW 预取 |
| Qualcomm Oryon (Snapdragon 8 Elite) | `oryon-1`, `oryon-2` | ARMv9.2 + SVE2 + SME，P+E 异构，启用 SW 预取 |
| Arm Neoverse N1/N2/V1 | `neoverse-n1/n2/v1` | 服务器核心，映射到 Ampere 策略 |
| Arm Cortex-X4 / Apple M4 | `cortex-x4`, `apple-m4` | 高 IPC 大核，映射到 Oryon 策略 |

---

## 使用方法

```bash
# 构建插件（需要 LLVM 18+ dev 头文件）
cmake -B build -DLLVM_DIR=$(llvm-config --cmakedir) .
cmake --build build

# 手动挂载（Oryon 目标）
clang -mcpu=oryon-1 -O2 -fpass-plugin=./AArch64HeteroOpt.so your_code.c -o out

# 独立测试（不需要 LLVM）
make test
make test_asan   # AddressSanitizer
make test_ubsan  # UBSanitizer
```

---

## 预取策略

### 预取局部性决策表

| 访问语义 | Oryon locality | Ampere locality | 说明 |
|----------|---------------|-----------------|------|
| HighReuse（含外层循环的内层，大步长） | 3 (PLDL1KEEP) | 2 (PLDL2KEEP) | Oryon 拉入 L1，Ampere 限于 L2 |
| ModerateReuse（一般步长） | 2 (PLDL2KEEP) | 2 (PLDL2KEEP) | 两者均使用 L2 |
| Streaming（小步长，无外层循环） | 1 (PLDL1STRM) | -1（跳过） | Ampere HW 预取负责流式访问 |
| Invariant（循环不变地址） | -1（跳过） | -1（跳过） | 已在 L1 中 |
| Unknown（变步长/SCEV 失败） | -1（跳过） | -1（跳过） | 无法预测 |

### 预取预算（避免 AGU/ICache 压力）

| 目标 | 每函数预算 | 每循环预算 | 最大预取距离 |
|------|-----------|-----------|------------|
| Ampere | 16 | 4 | 4096 B |
| Oryon  | 24 | 6 | 8192 B |

---

## 已知问题与缓解措施

### 问题 1：LLVM SVE tail-folding 死循环（README 原描述）

**根因**：`LoopVectorizer` 在 AArch64 + SVE 上对 low/dynamic trip count 的处理：

- 向量化器对小循环（TC 很小）生成 `vector body + active lane mask`
- 当 VF（`vscale`）和运行时 TC 不匹配时：
  - active lane mask 控制的循环可能永远不退出
  - 或 mask 掩码覆盖不到所有 iterations
- 对于动态 TC，SCEV 分析不够精确 → backedge taken count 无法保证 → 死循环

**Ahp 缓解措施（保守 workaround）**：

当 `TargetProfile::guard_sve_tail_fold == true`（Oryon/SVE2 目标启用）时，对 **SCEV 无法证明常量 backedge-taken count** 的循环跳过所有预取注入。

```cpp
if (Profile_.guard_sve_tail_fold) {
  const SCEV *BTC = SE.getBackedgeTakenCount(&L);
  if (isa<SCEVCouldNotCompute>(BTC)) {
    // 跳过此循环，避免与 tail-folding mask 交互
    return PreservedAnalyses::all();
  }
}
```

**长期方案**：需要 LLVM 上游彻底修复 AArch64 LoopVectorizer 对 SVE tail-folding 在动态 TC 场景下的重构。短期内无法从 pass 层面完全解决。

---

## 构建系统选项

| CMake 选项 | 说明 | 默认 |
|------------|------|------|
| `USE_SANITIZERS` | `address` / `undefined` / `thread` | 关 |
| `ENABLE_LTO` | 开启链接时优化 | 关 |
| `CLANG_TIDY` | clang-tidy 可执行路径（自动检测） | 自动 |

---

## CI 矩阵

| Job | Runner | 说明 |
|-----|--------|------|
| `standalone` | ubuntu-24.04 (x86-64) | 无 LLVM 独立测试 + ASan + UBSan |
| `llvm-plugin` | ubuntu-24.04-arm | Release/Debug/RelWithDebInfo + clang-tidy |
| `sanitized-plugin` | ubuntu-24.04-arm | ASan / UBSan 插件构建 |
| `build-and-release` | ubuntu-24.04-arm | tag push → GitHub Release |

---

## 版本

当前版本：`0.4.0`（见 `AArch64Features.h` 中的 `AHP_VERSION_*` 宏）

---

## 许可证

Apache-2.0 WITH LLVM-exception（见 `LICENSE`）
