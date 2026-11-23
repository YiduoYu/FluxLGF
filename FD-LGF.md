# FD-LGF：虚拟区域格点 Green 函数预条件（快速对角化背景）

**名字。** FD-LGF = **F**ictitious-**D**omain **L**attice **G**reen's **F**unction preconditioner with an FDM background，
中文“虚拟区域格点 Green 函数预条件”。FDM 指快速对角化方法（fast diagonalisation method）。
H30–H32 的报告里叫它“方法二”，从 2026-11-03 起统一用 FD-LGF 这个名字。

本文件夹包含 FD-LGF 的独立实现和全部对比实验，是 2026-11-03 从 `CNN function/axy_lgf/autoresearch_lgf_amg/method2_fdlgf` 移过来的。
研究背景见 `../CNN function/axy_lgf/autoresearch_lgf_amg/findings.md`（H7c、H21、H22、H30–H32）。
代码依赖 `../CNN function/axy_lgf` 包里的网格、算子和 PCG，脚本会自动把 `../CNN function` 加进 `sys.path`。

## 1. 问题
在 Ω = (0,1)² 上解 −∇·(a∇u) + q u = f，边界条件 u = 0。离散用项目统一的 5 点通量格式（三维是 7 点），
面系数取两端节点的算术平均（与 Xia 论文式 (5) 相同），得到对称正定系统 A u = f。

## 2. FD-LGF 的详细说明

### 2.1 出发点：传统 LGF 为什么在高对比度下失效
传统 LGF（Xia 论文式 (14)–(15)）用 M = ā(−Δ_h) + λI 作预条件，其中 ā = mean(a)，用 DST 精确求逆。
M 和 A 的差 A − M 分布在整个区域，量级与 a 的变化幅度相同，所以 PA 的条件数随对比度 χ 增长。
实验里 χ = 10⁴ 时就需要上千次迭代。快速变换（DST/FFT）能精确求逆的只是 *常系数* 或 *可分离系数* 的算子，
传统 LGF 的问题是：系数里能被精确求逆的那部分，它也用一个常数近似掉了。

### 2.2 核心思想：能精确求逆的部分精确求逆，剩下的局部精确求解
把系数分成两部分：

    a(x, y) = b(x, y) · φ(x, y),   b = a1(x) · a2(y)（可分离的“背景”），  φ 只在“特征”（夹杂、通道等）内 ≠ 1。

- **背景**交给快速变换：系数为 b 的通量算子 B 可以 *精确* 快速求逆，b 本身可以有 10⁸ 的对比度。
- **特征**交给局部直接解：A 和 B 只在特征附近不同，就在那里用 A 自己的子矩阵精确求解。
- **虚拟区域**的意思是：G = B⁻¹ 定义在整个方盒上，好像特征不存在、被背景系数“填满”了一样。

这两个部件都是精确的，所以对比度 χ 不进入任何近似误差。剩下的误差只来自几何：局部区域和背景区域如何衔接。

### 2.3 背景逆 G（“FDM 背景”）
b = a1(x)a2(y) 时，x 方向的面系数是 a2(y_j)·½(a1(x_i)+a1(x_{i+1}))，y 方向同理，所以背景算子可以精确写成张量积形式：

    B U = D2 U Tx + Ty U D1 + (q 的可分离部分)，  D1 = diag(a1)，D2 = diag(a2)，Tx、Ty 是一维通量三对角矩阵。

这里 U 是网格函数排成的矩阵，行对应 y、列对应 x。解两个一维广义特征值问题 Tx V = D1 V Λ、Ty W = D2 W M，就得到

    B⁻¹ F = W [ (Wᵀ F V) ./ (μ_j + λ_i) ] Vᵀ        （Lynch–Rice–Thomas 1964 的快速对角化）

按背景的类型分三种实现：
| 背景 | 实现 | 每次作用的代价 |
|---|---|---|
| a = a1(x)（例如教授的 χ^x、阶跃层） | y 方向 DST-I，每个 y 模态在 x 方向解三对角（`DSTTridiagBackground`） | O(n² log n) |
| a = a2(y) | 转置后同上（`TransposedBackground`） | O(n² log n) |
| 一般 a1(x)a2(y) | 上面的稠密快速对角化（`FDMBackground`） | O(n³)（二维） |

反应项 q 用它的可分离部分代替（对 y 取平均），这是 G 中唯一的近似；如果 q 本身只依赖 x，G 就是 B 的精确逆。
没有特征时 FD-LGF 就只有 G，也就是 H19/H22 的“广义 LGF”（方法一）。

### 2.4 局部精确解 Q
- 特征集：{ |log a − log b| > log 2 }，也就是 a 和背景相差超过 2 倍的节点。
- 局部区域：特征集向外膨胀一圈物理宽度为 w 的“环”（默认 w = 1/16，按网格换算成 ⌈w(n+1)⌉ 个格点），再按连通分量分块。
- 每块取 A 的对应子矩阵 A_k（区域外的值视为 0），做一次稀疏 LU，得到 Q = Σ_k R_kᵀ A_k⁻¹ R_k。
- 局部解必须精确。H9 试过每块只做一个 AMG V-cycle 或 Chebyshev 多项式，迭代数会变成 25–540 次。

### 2.5 平衡组合，以及它为什么对称正定
一次预条件作用相当于对称的逐次子空间校正（Xu 1992 意义下的对称乘法 Schwarz）：先局部校正，再全局背景校正，最后再局部校正。
误差传播算子是 E = (I − QA)(I − GA)(I − QA)。由于局部解精确，QAQ = Q，展开后得到

    P = (I − E) A⁻¹ = Q + (I − QA) G (I − AQ)

G 和 A_k 都对称正定，所以 P 对称正定，可以直接用 CG。这个形式和 BNN / deflation 的平衡预条件（Mandel 1993；Tang 等 2009）相同，
但两个角色正好反过来：那里的 Q 是低维的 *粗空间*，这里的 Q 是 *局部* 精确解；那里的 M 是局部或光滑子，这里的 G 是 *全局* 的精确背景逆。

### 2.6 为什么迭代数与 χ、n 无关
- G 对背景精确，A_k⁻¹ 对局部精确，两者都不随 χ 变差。
- 收敛速度由两个子空间（局部区域，以及其余部分上的背景）之间的“夹角”决定，即强化 Cauchy–Schwarz 常数。
  环越宽，局部区域把 A ≠ B 的地方包得越严实，夹角越大，迭代越少。H9 中 w = 1/64 → 1/16 时迭代从约 17 次降到 9 次。
- 环宽按 *物理* 宽度固定时，网格加密不会改变这个几何关系，所以迭代数与 n 无关（二维 n = 127–1023、三维 n = 31–127 都已验证）。

### 2.7 代价
- setup：背景的三对角分解（或两个 n×n 特征值问题），加上每个局部区域的稀疏 LU。
- 每次作用：1 次 G（DST + 三对角回代），2 次 Q（LU 回代），2 次 A 乘向量。
- 总代价主要由局部区域的大小（局部比例）决定。二维 SuperLU 的代价约为 N_loc^1.5；三维时填充增长得更快。

### 2.8 适用范围
| 情况 | 表现 |
|---|---|
| 可分离背景（只依赖 x，或 a1(x)a2(y)），没有特征 | 1–3 次迭代，最快 |
| 可分离背景 + 体积较小的紧凑特征（夹杂、通道、墙） | 迭代数与 χ、n 无关；时间取决于局部比例 |
| 特征占区域一半左右（棋盘、随机两相、随机场） | 局部区域几乎覆盖整个网格，FD-LGF 退化为一次稀疏直接解（1 次迭代，但 setup 就是整体 LU） |
| 背景不可分离且光滑（旋转分层、一般随机场） | 可分离拟合不好，特征无处不在，同上 |

### 2.9 与其他方法的区别
详见 `literature.md`。最接近的是电容矩阵法（Buzbee 等 1971；Proskurowski–Widlund 1976）、
fat boundary method（Maury 2001）、AGKS 高/低系数分块（Aksoylu 等 2008），以及 BNN/deflation 平衡形式（Mandel 1993）。

## 3. 文件
| 文件 | 内容 |
|---|---|
| `fdlgf.py` | FD-LGF 实现：`split_coefficient`、`DSTTridiagBackground`、`TransposedBackground`、`FDMBackground`、`FDLGFPreconditioner`、`solve_fdlgf_pcg` |
| `cases.py` | 二维算例 A–D（χ^x、阶跃层、χ^x+规则夹杂、χ^x+随机夹杂） |
| `high_contrast/` | 不只依赖 x 的高对比度系数族（`cases_hc.py`）及对比脚本（`bench_hc.py`），没有传统 LGF |
| `three_d/` | 三维版本：算子、四种方法、算例、基准测试 |
| `baselines/` | 对比方法：`gmg.py`（Jacobi 阻尼取教科书值）、`boomeramg.py`（hypre）、`lgf.py`（传统 LGF，只用于 H31/H32，以后的实验不再使用） |
| `benchmark.py`、`make_figures.py`、`report/` | H31 二维对比、作图、LaTeX 报告 |
| `test_fdlgf.py`、`test_baselines.py`、`three_d/test_3d.py` | 测试 |
| `protocol.md`、`three_d/protocol3d.md`、`high_contrast/protocol_hc.md` | 实验前锁定的方案 |
| `literature.md` | 文献对比 |

## 4. 用法
```python
from fdlgf import solve_fdlgf_pcg
res = solve_fdlgf_pcg(a, f, grid, q=q, a_nodes=a_nodes, tol=1e-10)                  # 自动拆分背景/特征
res = solve_fdlgf_pcg(a, f, grid, q=q, a_nodes=a_nodes, split=(a1, a2, mask), w=1/64)  # 指定背景、特征和环宽
```
`a_nodes` 是包含边界节点的 (n+2)×(n+2) 系数数组。自动拆分用交替中位数拟合 log a ≈ g1(x) + g2(y)，
要求特征在每一行、每一列中都占不到一半的节点；不满足时请自己传入 `split`。

运行（BoomerAMG 需要 hypre_env；BLAS 设成单线程，计时才公平）：
```bash
cd "Nonlinear Poisson/FD_LGF/high_contrast"
OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /opt/anaconda3/envs/hypre_env/bin/python bench_hc.py NAME 255 511
```
```bash
/opt/anaconda3/envs/torch_env/bin/python -m pytest -q test_fdlgf.py test_baselines.py three_d/test_3d.py
```

## 5. 结果
`RESULTS.md`、`report/report.pdf`（H31/H32），高对比度系数族的结果见 `high_contrast/`。
