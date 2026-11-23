# FD-LGF（原“方法二”）与相关方法的文献对比

检索时间 2026-11-03。多数条目只读了摘要或二手引用，引用前请核对原文。
方法二的定义见 `README.md`：P = Q + (I − QA) G (I − AQ)，G 是分层背景 b = a1(x)a2(y) 的精确快速逆，
Q 是在“夹杂 + 一圈宽度为 w 的环”上的精确稀疏 LU 局部求解。

## 1. 方法二由哪几块组成，每块的来源

| 组成部分 | 来源 | 方法二的改动 |
|---|---|---|
| 用快速直接解法（FFT/DST）做预条件 | Concus & Golub (1973) 用可分离算子的快速解法作非可分离问题的预条件 | 背景用 a1(x)a2(y) 的**精确**逆，不再是常系数 |
| 可分离算子的快速对角化 | Lynch, Rice & Thomas (1964)，Numer. Math. 6:185–199 | a2 为常数时改成 y 方向 DST + x 方向三对角，O(n² log n) |
| 把整体快速解法和局部修正组合起来 | 电容矩阵法 Buzbee, Dorr, George & Golub (1971)；Proskurowski & Widlund (1976)；Maury 的 fat boundary method (2001) | 局部问题放在“特征 + 环”的子区域上精确求解，用于有限系数的夹杂，不只是孔洞 |
| 平衡（balancing）形式 Q + (I−QA)M(I−AQ) | Mandel (1993) BNN；Tang, Nabben, Vuik & Erlangga (2009) 的比较 | 两个角色互换：Q 是**局部**精确解，G 是**全局**细网格快速解，不需要粗空间 |
| 按高/低系数分块的思路 | Aksoylu, Graham, Klie & Scheichl (2008) | 分块按“背景 / 特征”而不是“高 / 低”来分，背景本身可以有 1e8 的分层对比度 |

## 2. 与各类方法的区别

### 2.1 传统 LGF（常系数参考介质）
- 本项目的传统 LGF 是常系数 a0 = mean(a) 的 Dirichlet 格点 Green 函数，用 DST 作用。
  它和 FFT 均匀化中的“参考介质”同类：Moulinec & Suquet (1998) 的基本格式迭代数随对比度增长，
  Zeman 等 (2010) 报告用 CG 加速后迭代数仍约与 √(对比度) 成正比。
- 方法二把对比度放进两个**精确**的部件：分层部分在 G 里精确求逆，夹杂部分在 Q 里精确求逆。
  所以迭代数不随 χ 增长。代价是需要知道（或能拟合出）背景和夹杂的位置。
- 注意：计算流体中的 LGF（Liska & Colonius, 2016）是无界区域上常系数格点 Green 函数配快速多极，
  与本项目的“有界盒子 + DST”不是同一种实现，但同样只适用于常系数算子。

### 2.2 可分离 / 快速对角化预条件（本项目的方法一）
- Concus–Golub、Lynch–Rice–Thomas、Huynh–Ito–Toivanen（分层介质的可分离预条件）都只能精确处理
  a = a1(x)a2(y)。遇到不可分离的夹杂，它们只能作为近似，迭代数随夹杂对比度上升（H30 中方法一在算例 C
  上要 343 次以上或不收敛）。
- 方法二 = 方法一 + 在夹杂附近做精确局部修正。没有夹杂时两者完全相同。

### 2.3 电容矩阵法与 fat boundary method（最接近的经典方法）
- 电容矩阵法（Buzbee 等 1971；Proskurowski & Widlund 1976）：把不规则区域嵌入矩形，用快速 Poisson
  解法求解，再在不规则边界上解一个小的电容（Schur 补）方程。
- Maury (2001) 的 fat boundary method：整体问题在全区域上用快速解法，局部问题放在每个孔洞连通分量
  周围的窄带里，两者用带松弛参数 θ 的不动点迭代耦合。**结构上和方法二最像**（全局快速解 + 窄带局部问题）。
- 区别：
  1. 这两类方法处理的是孔洞或边界（局部问题上加 Dirichlet 条件）；方法二处理系数有限、可高可低的夹杂，
     夹杂内部的方程照常保留。
  2. 方法二是纯代数的：Q 直接取刚度矩阵 A 在子区域上的子块，不需要重新建立界面条件或有限元离散。
  3. 方法二是对称正定的预条件，放在 CG 里使用，没有需要调的松弛参数；fat boundary method 是不动点迭代，
     收敛依赖 θ。
  4. 方法二的整体快速解法针对的是变系数的分层背景，而不是常系数 Poisson。

### 2.4 两层区域分解、BNN、deflation、GenEO
- 形式相同：Q + (I − QA) M (I − AQ)。但在 BNN / deflation 中，Q 是**粗空间**修正（维数小），M 是局部或
  光滑子；要对高对比度稳健，粗空间必须包含与系数相关的基（Graham, Lechner & Scheichl 2007；GenEO,
  Spillane 等 2014；Galvis–Efendiev），否则条件数随对比度增长。
- 方法二里，Q 是局部精确解（维数 = 夹杂 + 环 的节点数，约占 25%–50%），G 是全局、全尺度的精确背景逆。
  所以它不需要粗空间，网格无关性来自 G 在所有尺度上都对背景精确。
- 用 Xu (1992) 的子空间校正框架来看，方法二是两个子空间的对称逐次校正（对称乘法 Schwarz）：
  子空间 1 = 特征 + 环（精确求解），子空间 2 = 全空间（用背景算子近似求解）。环越宽，两个子空间的重叠越大，
  迭代数越少，但 LU 越贵（见 H9 和本次的 w 对比）。

### 2.5 GMG 与 BoomerAMG
- 几何多重网格用双线性插值和 Jacobi 光滑，系数间断不和粗网格对齐时会退化。Alcouffe, Brandt, Dendy
  & Painter (1981) 用依赖算子的插值（black-box MG）来修复这一点。本项目的 GMG 是标准的几何插值 + Galerkin
  粗算子，所以在 C、D 上迭代数明显变多。
- BoomerAMG（Henson & Yang 2002，hypre）从矩阵自动构造粗网格和插值，是“黑箱”方法：不需要知道系数结构，
  对各种对比度都稳健，并且可以并行。
- 方法二是“灰箱”方法：需要笛卡尔网格、可分离的背景、以及能识别出的夹杂。在这些条件满足时，
  它的迭代数和 χ、n 都无关，而且背景部分只需 O(n² log n)。条件不满足时（例如背景是非可分离的光滑场），
  它要么不能用，要么退化成近似的可分离预条件。

### 2.6 Green–Jacobi FFT 预条件（Ladecký 等 2025）
- 针对光滑的高对比度系数，用 J^{1/2} G0 J^{1/2}（J = diag(A)^{-1}）的缩放；作者把尖锐界面的理论留作以后的工作。
- 方法二针对的是分层背景 + 不连续夹杂；两者适用的系数类型互补。

### 2.7 层次化 / 直接解法
- 嵌套剖分 LU、层次插值分解（HIF）等对整个问题做直接分解，鲁棒但二维时存储约 O(N log N)–O(N^{1.5})。
  方法二只对夹杂附近做直接分解，其余部分用快速变换。

## 3. 小结：方法二的定位
- 据目前检索，没有找到把“分层背景的精确快速逆（DST + 三对角 / 快速对角化）”和“夹杂附近的精确局部解”
  放进同一个平衡型 SPD 预条件、并在 χ 到 1e8 的情况下与 AMG 做对比的文献。最接近的是电容矩阵法、
  fat boundary method、AGKS 和 BNN，见上面的区别。
- 优点：迭代数与 χ、n 无关；背景部分没有 setup 成本，作用一次很便宜；误差下降很快。
- 局限：需要结构信息（背景可分离、夹杂可识别、笛卡尔网格）；夹杂多或环宽时 LU 的 setup 随 n 超线性增长；
  目前是串行实现，BoomerAMG 有成熟的并行版本。

## 参考文献（含链接）
- Alcouffe, Brandt, Dendy, Painter (1981), SIAM J. Sci. Stat. Comput. 2:430–454. https://portal.mardi4nfdi.de/entity/Q790595
- Aksoylu, Graham, Klie, Scheichl (2008), Comput. Visual. Sci. 11:319–331. https://people.bath.ac.uk/masrs/agks_preprint_08.pdf
- Buzbee, Dorr, George, Golub (1971), SIAM J. Numer. Anal. 8(4):722–736, doi:10.1137/0708066.
- Concus, Golub (1973), SIAM J. Numer. Anal. 10:1103–1120.
- Graham, Lechner, Scheichl (2007), Numer. Math. 106:589–626. https://people.bath.ac.uk/masrs/gra_lec_sch_preprint_1106.pdf
- Henson, Yang (2002), BoomerAMG, Appl. Numer. Math. 41:155–177. https://hypre.readthedocs.io/en/latest/solvers-boomeramg.html
- Huynh, Ito, Toivanen, DD16. https://www.ddm.org/DD16/homepage/cims.nyu.edu/dd16/proceedings/toivanen.pdf
- Ladecký 等 (2025), arXiv:2508.02613. https://arxiv.org/abs/2508.02613
- Liska, Colonius (2016), J. Comput. Phys.（LGF 浸入边界）. https://ar5iv.arxiv.org/html/1604.01814
- Lynch, Rice, Thomas (1964), Numer. Math. 6:185–199, doi:10.1007/BF01386067.
- Mandel (1993), Balancing domain decomposition, Comm. Numer. Methods Eng. 9:233–241, doi:10.1002/cnm.1640090307.
- Maury (2001), A fat boundary method for the Poisson problem in a domain with holes, J. Sci. Comput. 16:319–339.
  https://link.springer.com/article/10.1023/A:1012821728631
- Moulinec, Suquet (1998), FFT 均匀化；Zeman 等 (2010) CG 加速. https://arxiv.org/abs/1004.1122
- Proskurowski, Widlund (1976), Math. Comp. 30:433–468. https://www.ams.org/mcom/1976-30-135/S0025-5718-1976-0421102-4/
- Spillane 等 (2014), GenEO, Numer. Math.
- Tang, Nabben, Vuik, Erlangga (2009), J. Sci. Comput.（deflation / BNN / 多重网格两层预条件比较）。
- Xu (1992), SIAM Review 34(4):581–613, doi:10.1137/1034116.

---

## 4. 补充（2026-11-03）：专门针对高对比度系数的方法

下文用 FD-LGF 指原“方法二”。各条只读了摘要或二手资料，引用前请核对原文。

### 4.1 谱粗空间 / 多尺度区域分解（Galvis–Efendiev；GenEO；Scheichl–Vassilevski–Zikatanov）
- **Galvis & Efendiev (2010), Multiscale Model. Simul. 8:1461–1483**：两层区域分解。粗空间加入局部加权特征值问题中小特征值对应的特征函数，
  这些特征函数正好对应每个孤立的高导夹杂和通道。可以证明条件数与对比度无关，代价是夹杂和通道越多，粗空间越大；
  同年还有一篇论文专门讨论如何降低粗空间维数（8:1621–1644）。
  https://isc.tamu.edu/resources/preprints/2009/2009-03.pdf
- **Efendiev, Galvis, Lazarov, Willems (2012), ESAIM M2AN 46**：推广到一般对称正定双线性形式。https://ar5iv.labs.arxiv.org/html/1105.1131
- **Scheichl, Vassilevski, Zikatanov (2011), Multiscale Model. Simul. 9:1677–1699**：带泛函约束的能量最小粗空间（AMGe / 重叠 Schwarz），
  逼近和稳定性常数与系数变化无关。https://epubs.siam.org/doi/10.1137/110821639
- **与 FD-LGF 的区别**：这类方法让“粗空间”去认识每个夹杂和通道（一个夹杂或通道对应一个或几个粗基函数），
  所以需要网格划分、局部特征值问题，以及可能很大的粗空间。FD-LGF 没有粗空间：背景的全尺度信息由精确的快速逆 G 负责，
  夹杂由局部 LU 负责。代价转移到了 LU 上：夹杂体积越大，LU 越贵；这一点和“夹杂个数越多，粗空间越大”正好对应。

### 4.2 鞍点重写（Gorb, Kurzanova, Kuznetsov 2018）
- arXiv:1801.01578、arXiv:1808.00627：把高导夹杂问题写成等价的鞍点系统，再给出对 Lanczos/MINRES、Uzawa、PCG 稳健的预条件，
  迭代数与对比度和网格无关；测试对象是分布在区域中的高导颗粒。Kurzanova (2017) 的博士论文还讨论了无穷对比度和绝缘夹杂。
  https://arxiv.org/abs/1801.01578 · https://arxiv.org/abs/1808.00627
- **与 FD-LGF 的区别**：它们改的是问题的代数形式（加入拉格朗日乘子），FD-LGF 不改问题，只改预条件。
  两者都利用了“夹杂是局部的”这一点；FD-LGF 还额外要求背景可分离。

### 4.3 FFT / Green 函数类（Ladecký, Pultarová, Zeman 等）
- **Ladecký 等 (2023), Appl. Math. Comput. 446**：用周期均匀参考问题的离散 Green 函数作有限元 FFT 求解器的预条件，是最优的。
- **Ladecký 等 (2025), arXiv:2508.02613**：Green–Jacobi 预条件，针对 *光滑* 高对比度系数；文中说明对尖锐界面效果不好。
- **与 FD-LGF 的区别**：参考介质始终是常系数（只是做了缩放）。FD-LGF 的参考介质是分层的（可分离的），再加上局部精确修正，
  所以能处理 χ^x、阶跃层和带夹杂的分层介质；但它要求背景可分离，而 Green–Jacobi 可以用于任何光滑系数。

### 4.4 文献中的高对比度测试问题
- 夹杂/颗粒（Gorb–Kuznetsov 小组；Galvis–Efendiev）、高导通道（Galvis–Efendiev；Graham–Lechner–Scheichl）、
  棋盘/Kellogg 交叉点（对比度到 10⁸，arXiv:2202.08830、2210.10415）、高方差对数正态随机场（max/min 可达 8×10⁹，arXiv:math/9809043）、
  SPE10 储层渗透率（对比度约 10⁷，arXiv:2103.08513）。
- 本次新加的系数族（`high_contrast/cases_hc.py`）覆盖了以上各类：可分离二维、背景加通道、4×4 棋盘、对数正态、
  各向异性分层随机场（类 SPE10）、斜条纹、绝缘墙迷宫、渗流阈值处的随机两相介质，以及细条纹和粗糙随机场。

### 4.5 文献中 AMG 的弱点
- 经典（Ruge–Stüben）AMG 的强连接判据建立在 M-矩阵假设上。文献中公认的困难情形有：
  不与网格对齐的各向异性扩散（arXiv:1409.4702、arXiv:1106.5990），以及系数跳跃大到使强连接图断成多块、插值无法跨越的情形（arXiv:1611.01917）。
- 对于标量系数、5 点或 7 点通量格式（矩阵总是 M-矩阵）的跳跃系数问题，大量实验表明经典 AMG 和光滑聚集 AMG 都很稳健（arXiv:1611.01917 的综述）。
  本次二维扫描也得到同样的结论：10 类系数、χ 到 10⁸ 时，BoomerAMG 都只要 7–12 次迭代。
  因此要找 BoomerAMG 的弱点，应该看三维（迭代数随 n 增长）和强连接图被割裂的情形（绝缘墙、薄层）。

参考链接：
[arXiv:2202.08830](https://arxiv.org/pdf/2202.08830) · [arXiv:2210.10415](https://arxiv.org/pdf/2210.10415) ·
[arXiv:math/9809043](https://arxiv.org/pdf/math/9809043) · [arXiv:2103.08513](https://arxiv.org/pdf/2103.08513) ·
[arXiv:1611.01917](https://arxiv.org/pdf/1611.01917) · [arXiv:1409.4702](https://arxiv.org/pdf/1409.4702) ·
[arXiv:1106.5990](https://ar5iv.arxiv.org/html/1106.5990) · [Ladecký 2023](https://ideas.repec.org/a/eee/apmaco/v446y2023ics0096300323000048.html)

## 5. 补充（2026-11-03）：各向异性张量系数，AMG 已知的困难情形

### 5.1 问题与离散
- 方程改为 −∇·(K∇u)+qu=f，K = χ e eᵀ + (I − e eᵀ) 是对称正定张量，沿 e 的传导率为 χ、垂直方向为 1，各向异性比就是 χ。
- 离散采用标准对称 9 点格式（能量形式：面上 K11、K22 取相邻节点的算术平均，交叉项 K12 用中心差分）。K = aI 时它就是原来的 5 点通量格式。
  只要 K12 ≠ 0，矩阵就有正的非对角元，不再是 M-矩阵。这正是经典 AMG 强连接判据失去理论依据的情形。

### 5.2 文献中 AMG 在这类问题上的表现
- 旋转各向异性扩散（θ 不与网格对齐，ε 小）是 AMG 文献的标准难题。
  最小二乘 AMG 的新工作（arXiv:2601.04112）报告 ε=1e-7 时，经典 AMG 的平均收敛因子约 0.86，光滑聚集约 0.94，他们的方法约 0.5。
- Root-node AMG（Manteuffel–Olson–Schroder–Southworth，arXiv:1610.03154）专门处理旋转各向异性。他们报告：完全各向异性时，收敛因子随问题规模缓慢增长；经过改进后的 root-node AMG 可以做到与 h 无关。
- 磁化等离子体中沿磁力线的热传导（各向异性比 10³–10⁹，方向随空间变化，与网格不对齐）是另一类典型难题：
  Wimmer–Southworth 等（arXiv:2301.13351）指出常规基于扩散的 AMG 在高各向异性下会失效；
  Krzysik 等（arXiv:2501.00938）证明重叠 Schwarz 光滑子对各向异性不稳健，并称"真实各向异性水平下的高效并行多重网格仍基本是开放问题"。
- 本次算例据此选取：网格对齐（对照）、30° 和 45° 常数旋转、圆形磁力线（circ），以及各向同性背景中的局部旋转各向异性圆盘（disk）和斜断层带（band）。

参考链接：
[arXiv:2601.04112](https://arxiv.org/pdf/2601.04112) · [arXiv:1610.03154](https://arxiv.org/pdf/1610.03154) ·
[arXiv:2301.13351](https://arxiv.org/pdf/2301.13351) · [arXiv:2501.00938](https://arxiv.org/pdf/2501.00938) ·
[Reusken, Schur-complement two-grid](https://www.igpm.rwth-aachen.de/Download/reports/reusken/ARpaper15.pdf)

## 6. 补充（2026-11-03）：三温辐射扩散矩阵与 FD-LGF 的改进依据

### 6.1 三温（3T）能量方程的线性系统
- SolverChallenge26_01 是三维三温能量方程在 128³ 结构网格上的有限体积 7 点离散，矩阵为 3×3 块结构：
  每个温度一个扩散块，加上同一单元内电子–离子、电子–辐射的交换耦合（从阶数和非零元数可以精确反推）。
- 这类系统的主流解法都是"物理量分块 + AMG"：
  PCTL 物理量粗化两层法（Xu、Mo、An 提出；αSetup-PCTL 是它的自适应版本）、
  AMG 块三角预条件（Shu、Liu、Xu、Yue、Li，AAMM 2021）、
  物理量 Schwarz 预条件（Yue、He、Xu、Shu、Wang，CiCP 2022）、
  JASMIN 上 ILU 与 BoomerAMG 的组合预条件（Yue、Xu、Shu 2017）、
  以及多群辐射扩散的 AMG 块预条件（Yue 等，CiCP 2021）。

### 6.2 改进 FD-LGF 用到的三条线索
- **Jacobi 加速的 FFT 预条件**（Ladecký、Pultarová、Zeman、Pastewka 等，arXiv:2508.02613）：
  把全局 Green 函数预条件与局部 Jacobi 缩放串联，保持 O(N log N)，对"光滑的高对比度"系数有效。
  我们把标量 Jacobi 换成逐单元 3×3 块缩放，使它能处理三温之间的强耦合。
- **按高系数区域的 deflation**（Vuik、Segal、Meijerink，JCP 1999；Vuik–Frank）：
  小特征值的个数等于不与 Dirichlet 边界相连的高传导区域个数；用这些区域上的常数向量做 deflation，迭代数与对比度无关。
- **高对比度子块近似为常系数求解**（Aksoylu、Graham、Klie、Scheichl，CVS 2008）：
  奇异摄动分析表明，对比度足够高时，子块的逆可以用常系数求解近似，这正是 LGF 擅长的部分。
- FD-LGF 的平衡形式 P = Q + (I−QA)G(I−AQ) 与 BNN、平衡 deflation 是同一结构（Tang、Nabben、Vuik、Erlangga 2009 对这类两层方法有系统比较）。
  因此改进只需把"特征区域上的精确 LU"换成"粗空间上的 Galerkin 求解"，整体框架保持不变。

参考链接：
[arXiv:2508.02613](https://arxiv.org/abs/2508.02613) · [Vuik–Segal–Meijerink 1999](https://diamhomes.ewi.tudelft.nl/~kvuik/papers/Vui99SM.pdf) ·
[Vuik 2001](https://homepage.tudelft.nl/d2b4e/papers/Vui01SMW.pdf) · [AGKS 2008](https://link.springer.com/doi/10.1007/s00791-008-0105-1) ·
[Shu 等 2021](https://global-sci.com/article/73007/algebraic-multigrid-block-triangular-preconditioning-for-multidimensional-three-temperature-radiation-diffusion-equations) ·
[Yue 等 2022](https://ojs.global-sci.org/index.php/cicp/article/view/7101) ·
[αSetup-PCTL](https://global-sci.com/article/79497/asetup-pctl-an-adaptive-setup-based-two-level-preconditioner-for-sequence-of-linear-systems-of-three-temperature-energy-equations) ·
[arXiv:2511.12726](https://arxiv.org/pdf/2511.12726)
