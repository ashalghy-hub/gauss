# SIMD 编程实验：高斯消去与 Gröbner 基计算

本仓库包含了并行程序设计 SIMD 实验的全部核心源代码。
代码涵盖了基础的普通高斯消去（串行、自动向量化、手工 AVX 优化）以及期末进阶探索的特殊高斯消去（Gröbner 基计算）。

## 代码文件说明

本仓库共包含 3 份核心 C++ 源代码文件，分别对应实验的不同测试与优化阶段：

### 1. `lu_auto.cpp` (基准测试与自动向量化)
- **用途**：作为实验性能对比的 Baseline。
- **内容**：包含了没有任何优化的纯串行普通高斯消去算法。
- **实验目标**：结合特定的编译参数，用于测试“纯串行”与“GCC编译器自动向量化”的性能差异。

### 2. `lu_avx.cpp` (手工 AVX 向量化与对齐策略)
- **用途**：基础要求（80%）的核心实现文件。
- **内容**：使用 256-bit AVX/AVX2 Intrinsics 手工优化的普通高斯消去算法。
- **实验目标**：代码内部实现了`LU_AVX_Unaligned`和`LU_AVX_Aligned`两个函数，通过内存边界预处理，对比了`_mm256_loadu_ps`与`_mm256_load_ps`在现代CPU微架构下的性能与指令开销差异。

### 3. `groebner.cpp` (进阶探索：特殊高斯消去)
- **用途**：期末研究探索（冲击满分加分项）的核心代码。
- **内容**：针对密码学中 Gröbner 基计算（GF(2) 有限域运算）编写的大规模稀疏矩阵消去程序。
- **实验目标**：摒弃传统浮点数组，设计了 `BitRow` 长位向量进行状态压缩。采用了 **“宏观 OpenMP 动态调度并发 + 微观 `#pragma omp simd` 向量化批量异或”** 的双层并行策略，并引入临界区双重检查锁避免数据竞态。

---

## 编译与运行说明

环境要求：支持 AVX2 指令集的 x86 CPU，以及支持 OpenMP 的 GCC 编译器。
在 Linux 或 Windows 终端下，推荐使用以下命令进行编译和运行：

```bash
测试 1：纯串行与自动向量化对比
# 纯串行编译
g++ lu_auto.cpp -O0 -o run_baseline
./run_baseline

# 自动向量化编译
g++ lu_auto.cpp -O3 -mavx2 -ftree-vectorize -o run_auto
./run_auto

测试 2：手工 AVX 对齐与不对齐策略
g++ lu_avx.cpp -O2 -mavx2 -o run_avx
./run_avx

测试 3：Gröbner 基计算 (多线程+SIMD)
g++ groebner.cpp -O3 -mavx2 -fopenmp -o run_groebner
./run_groebner
