# Ahp — AArch64 Hetero-Opt LLVM Pass

针对 AArch64 的 LLVM New Pass Manager 插件：按目标微架构决定**要不要**注入软件预取、注入到哪一级缓存、多远。

## 它做什么

两个 pass，两个扩展点：

| Pass | 扩展点 | 动作 |
|------|--------|------|
| `AArch64LaneHintPass` | `LoopOptimizerEndEP`（向量化之前） | 给 FP 归约循环注入 `llvm.loop.unroll.count` / `llvm.loop.interleave.count`，拆开累加器依赖链 |
| `AArch64HeteroOptPass` | `OptimizerLastEP`（向量化之后） | 只注入 `llvm.prefetch`，不加新 load，不改 loop metadata |

预取插在**所服务的 load 之前**（`IRBuilder` 建在 load 位置）。这是为了回避支配关系问题：向量化后指针 GEP 在循环体内，往 preheader 插会违反支配。

## 目标

| 目标 | CPU 字符串 | SW 预取 | 说明 |
|------|-----------|---------|------|
| Ampere Altra / AmpereOne | `ampere1`, `ampere1a` | 关 | 硬件预取强，流式访问交给 HW |
| Neoverse N1/N2/V1 | `neoverse-n1/n2/v1` | 关 | 映射到 Ampere 策略 |
| Qualcomm Oryon | `oryon-1`, `oryon-2` | 开 | P+E 异构，按 4 个 FP load unit 调参 |
| Cortex-X4 / Apple M4 | `cortex-x4`, `apple-m4` | 开 | 映射到 Oryon 策略 |
| 未知 CPU | 其它 | 关 | 不知道微架构就不猜，绝不默认开 |

> **Oryon 没有 SVE/SVE2，也没有 SME。** LLVM 自己的 `oryon-1` 特性表是 NEON / AES / SHA2 / SM4 加若干 fusion 开关。所以 profile 里 `march_str` 写的是 `"oryon-1"` 而不是一串 `+sve2+sme`——照后者编译出来的指令这条核心执行不了。

## 使用

```bash
# 构建插件（需要 LLVM 18+ dev 头文件）
cmake -B build -DLLVM_DIR=$(llvm-config --cmakedir) .
cmake --build build

# 挂载
clang -mcpu=oryon-1 -O2 -fpass-plugin=./AArch64HeteroOpt.so your_code.c -o out

# 独立测试（不需要 LLVM，34 项检查）
make test
make test_asan   # AddressSanitizer
make test_ubsan  # UndefinedBehaviorSanitizer
```

调试：`-mllvm -debug-only=aarch64-hetero-opt`（需要 debug 版 LLVM）。

## 预取策略

### 局部性

locality 值沿用 `llvm.prefetch` 的约定：3=PLDL1KEEP、2=PLDL2KEEP、1=PLDL1STRM、-1=不注入。

| 访问语义 | Oryon | Ampere | 判据 |
|----------|-------|--------|------|
| HighReuse | 3 | 2 | 含外层循环，大步长 → 跨外层迭代复用 |
| ModReuse | 3 | 2 | 64B < 步长 ≤ 512B，硬件预取器的盲区 |
| Streaming | 1 | -1 | 步长 ≤ 64B 且无外层循环，HW 预取已足够 |
| Invariant | -1 | -1 | SCEV 判定循环不变，已在缓存里 |
| Unknown | -1 | -1 | 步长算不出来，不猜 |

### 预算

| 目标 | 每函数 | 每循环 | 最大预取距离 | gather 每循环 |
|------|-------|-------|------------|--------------|
| Ampere | 16 | 4 | 4096 B | 2 |
| Oryon | 32 | 8 | 8192 B | 4 |

预取距离不是固定的 `lookahead × stride`，而是 `clamp(min(lookahead, TC/2) × stride, min_dist, max_dist)`。`TC/2` 防止短循环把预取打出数组尾部。

另有工作集守卫：若 `TC × |stride|` 小于 L1D 的一半，整个工作集本来就在缓存里，直接跳过。

## 它不做什么

- **不处理指针追逐**（链表、树、探测表）：`classifyLoad()` 只认仿射 SCEV，非仿射一律跳过。`test/test_verify.cpp` 里那几个指针追逐用例目前是**不被优化**的对照样例。
- **不碰已向量化的循环**：`isAnyVectorized()` 命中就整循环跳过，避免在 vector GEP 上插标量预取。
- **不改 loop metadata 于向量化之后**：`setLoopID` 只在 `LoopOptimizerEndEP` 用。

## 验证

```
$ make test
=== All standalone tests passed ===
```

独立测试不需要 LLVM：`test_syntax.cpp` 自带一个最小的 `llvm::StringRef` 桩。桩模式必须把 `AARCH64_HETERO_HAS_REAL_LLVM` 显式定义为 **0**（该宏的含义是"有真 LLVM 头文件可用"），否则在装了 LLVM 的机器上会和真头文件撞车——只有没装 LLVM 的机器才编得过，这个坑已经修掉。

### 验证插件本体（不依赖 cmake）

`LLVMConfig.cmake` 在部分发行版（如 Termux 的 `llvm` 包）里会引用缺失文件，configure 直接失败。单个 .cpp 用 `llvm-config` 编更省事：

```bash
clang++ -DHAS_LLVM_HEADERS=1 -Iinclude -fPIC -shared \
        $(llvm-config --cxxflags) -std=c++20 \
        lib/AArch64HeteroOpt.cpp -o AArch64HeteroOpt.so \
        $(llvm-config --ldflags) -L$PREFIX/lib -lLLVM-21
```

两个必须注意的点：

- `llvm-config --cxxflags` 自带 `-std=c++17`，所以 `-std=c++20` 必须写在它**后面**，否则 `std::string_view::starts_with` 不存在。
- 必须显式 `-lLLVM-21`。不链接的话 .so 里是未定义符号，`dlopen` 会失败在 `llvm::DebugFlag`（宿主 clang 不导出 LLVM 符号）。

加载验证（`-fno-vectorize` 让循环保持标量，否则会被 `isAnyVectorized` 整循环跳过）：

```bash
clang -mcpu=oryon-1 -O2 -fno-vectorize -fno-slp-vectorize \
      -fpass-plugin=./AArch64HeteroOpt.so -S -emit-llvm test/sweep.c -o sweep.ll
awk '/^define/{n=$0} /call void @llvm.prefetch/{print n}' sweep.ll | uniq -c
```

`test/sweep.c` 是同一段循环的步长扫描（8/16/64/72/128/520 字节）：前三者应得 locality 1，后三者得 locality 3。

CI：standalone（x86-64，ASan/UBSan）、llvm-plugin（arm，Release/Debug/RelWithDebInfo + clang-tidy）、sanitized-plugin、tag 发布。

## 实测

设备：骁龙 8 Elite（Oryon，无 SVE），`-O2 -mcpu=oryon-1`，矩阵 640×640 double。
`test/bench_matrix.c`，base 与插件版交替跑三轮，取每项最小值。

| 用例 | base | 加插件 | 变化 |
|---|---|---|---|
| `dgemm_ijk`（内层 k，B 步长 5120 B，标量 FP 归约） | 238.0 ms | 192.8 ms | **1.23×** |
| `dgemm_ikj`（内层 j，单位步长，已被向量化） | 41.3 ms | 42.7 ms | −3.4% |
| `spmv`（CSR，每行 5 个非零，列号 ×97 打散） | 2 µs/次 | 2 µs/次 | 噪声内，无结论 |

结论：**只有大步长、且没被向量化的循环真受益**（1.23×），这和 pass 的设计一致。单位步长的向量化循环本来就被 `isAnyVectorized` 跳过整循环，但它的外层/中层循环仍会被注入预取，代价是 3~4%。

把两边都加 `-fno-unroll-loops` 再比（排除 `LaneHintPass` 的 unroll 提示灌水）：238.98 → 194.02 ms，仍是 1.23×，说明**这点收益来自预取本身，unroll 提示没测出额外贡献**。

SpMV 那条测不出来是因为规模太小（3200 个非零 = 2 µs），要在有意义的规模上评估 gather 预取，得换一个百万级非零的矩阵。

## 版本与许可

当前 `0.4.0`（见 `include/AArch64Features.h` 的 `AHP_VERSION_*`）。

Apache-2.0 WITH LLVM-exception，见 `LICENSE`。
