# FD_LGF / regions 独立审计

审计日期：2026-11-05。原始结果快照：北京时间 21:38:48（H56 的 G11 当时只记录了 2/10 步）。审计不修改求解器、生成器或历史结果；新增文件全部在本目录。P1 表示应在对外使用关键结论前处理，P2 表示应修复或补充验证。

## 判断

现有证据支持一个有限结论：在所记录的两批真实矩阵实验中，bt-fd 用双双精度迭代修正达到了原矩阵相对残差 1e-8，耗时低于指定的 AMG3D 基线。没有找到该残差被算小若干数量级的证据；小规模精确核验支持其实现。

目前不能把它升级为“对调优后的 BoomerAMG 有已确认的固定加速比”“自报 1e-8 等于相同解精度”“约 16% 是通用胜负分界”，也不足以据跨批次耗时决定比赛一律用 Cholesky。数学主体通过了本次有限的独立代数检查，但找到一个明确边界实现缺陷和一个影响参数敏感性测试的配置缺陷。

## 1. [P1，确定] 相同自报阈值对应不同的实际残差，精度公平性需要重写

定位：[run_real.py:57](/Users/duoduo/Documents/温州肯恩大学/科研/Nonlinear Poisson/FD_LGF/regions/run_real.py:57)、[原始 H53 R1 表](/Users/duoduo/Documents/温州肯恩大学/科研/Nonlinear Poisson/FD_LGF/audit_20261105/evidence/original_snapshot/h53_R1.csv:1)。

同为 solver-reported rtol=1e-8：

| 方法 | 自报残差 | 双精度重算的原始残差 | 对角缩放残差 |
|---|---:|---:|---:|
| AMG3D | 4.80e-9 | 1.343e-2 | 3.849e-13 |
| bt-fd | 9.39e-9 | 2.399e-2 | 9.157e-10 |

bt-fd 的缩放残差约为 AMG 的 2,379 倍。rtol=1e-4 时对应约 44.6 倍。H55 的 bt-fdd 在自报 1e-8 时缩放残差约 2e-16，也比 bt-fd 严得多。因此“16 步对 60 步”本身不足以说明达到相同精度更高效。

PETSc 的 UNPRECONDITIONED 设置没有把 GMRES 的递推残差变成重新计算的 b-Ax；官方文档也明确该返回值可能只是近似量：[KSPGetResidualNorm](https://petsc.org/release/manualpages/KSP/KSPGetResidualNorm/)。报告把列名标成“自报”是准确的，问题在于据此延伸出的精度、优劣解释。缩放残差也不是解误差，不能从上述倍数推断解误差倍数。

建议：把自报阈值结果保留为诊断；主比较统一用同一高精度原矩阵残差，或明确选定缩放残差并真正以它验收。记录分场残差、最终解表示以及必要时的前向误差。对自报残差到阈值的时间，不再使用“同精度更快”的表述。

## 2. [P2，确定] AMG 设置覆盖调用者参数，常规参数试验可能根本没改到参数

定位：[run_ir.py:32](/Users/duoduo/Documents/温州肯恩大学/科研/Nonlinear Poisson/FD_LGF/regions/run_ir.py:32)、[run_real.py:47](/Users/duoduo/Documents/温州肯恩大学/科研/Nonlinear Poisson/FD_LGF/regions/run_real.py:47)、[seq_lock.py:52](/Users/duoduo/Documents/温州肯恩大学/科研/Nonlinear Poisson/FD_LGF/regions/seq_lock.py:52)、[bt_elim.py:18](/Users/duoduo/Documents/温州肯恩大学/科研/Nonlinear Poisson/FD_LGF/solver_challenge/bt_elim.py:18)。

这些位置无条件向 PETSc.Options 写入 AMG3D。独立复现：先将 ir_amg_pc_hypre_boomeramg_strong_threshold 设为 0.7，调用生产 inner() 后数据库值与 ksp.view() 均为 0.5。[完整输出](/Users/duoduo/Documents/温州肯恩大学/科研/Nonlinear Poisson/FD_LGF/audit_20261105/evidence/numeric.log:1)。因此从预置 options 或环境注入参数时，不能依赖 setFromOptions 保留它们。

修复：默认值只在选项不存在时写入，或者接收显式的配置对象；把有效的 ksp.view()/options 和库版本写入每次结果，避免只记录期望参数。CLI 参数需正确交给 petsc4py 初始化并与方法列表分开解析。

这组基线本身没有明显不合理之处。HMIS、ext+i、Pmax=4 和适度 aggressive coarsening 属于 hypre 推荐组合：[BoomerAMG 文档](https://hypre.readthedocs.io/en/latest/solvers-boomeramg.html)。本机实际为一轮 V-cycle、tol=0、对称 SOR/Jacobi；不能把 hypre 文档中的所有默认平滑器设置直接当成 PETSc 包装器的实际设置。

也不能说项目从未调过 AMG：[generalize/diag.py:165](/Users/duoduo/Documents/温州肯恩大学/科研/Nonlinear Poisson/FD_LGF/generalize/diag.py:165) 已有 36 组扫描，旧的 3t-base 结果中 theta=0.25 为 1.595 s、基线为 1.707 s。不过该扫描针对 m=64 合成矩阵和不同判据，不能替代真实矩阵、原残差 IR 及 H56 复用目标下的调优。

### 本次独立参数扫描

使用三个合成问题：m=32 单胶囊、m=32 十胶囊、m=64 单胶囊；均 delta=0.02。每题 12 组 AMG 加 bt-fd，各做两遍，共 78 次。独立进程、固定随机顺序、限制线程；AMG 覆盖 theta=0.25/0.5/0.7、agg=0/1/2、Pmax=2/4/8，以及一组 PMIS。所有问题采用 GMRES(50)、内层 1e-6、双双精度原残差 1e-8，全部通过。bt-fd 用 H56 的 tol=0.1。此扫描不是对 H53 默认匹配容差的原样复现。

| 小规模问题 | 原 AMG 基线 / s | 扫描中最快 AMG / s | bt-fd / s | bt-fd / 最快 AMG |
|---|---:|---:|---:|---:|
| m=32，单胶囊 | 0.309 | 0.305 | 0.187 | 0.613 |
| m=32，十胶囊 | 0.382 | 0.327 | 0.443 | 1.356 |
| m=64，单胶囊 | 4.067 | 3.550 | 2.274 | 0.640 |

两个 m=32 问题最快观察配置是 HMIS/theta=0.25/agg=1/Pmax=4，m=64 为 theta=0.7，其余相同。调参没有逆转这三个问题的胜负；部分用例 AMG 观察时间降低约 13–15%。但仅两遍，而且审计期间原 H56 大任务仍在更新日志、系统存在内存压力；这些是探索性时间，不能宣称找到了统计可靠的最优参数。m=64 基线两遍也有明显波动。迭代数、达标与否比这些秒数更适合本次核查。

真实 SC26_01 的“换参数后是否仍更快”仍未获得本次实测答案；不把小规模结果外推到 629 万未知数。下一次正式比较应单独隔离真实矩阵作配对重复，分别优化单次 setup+solve、IR 总时间、冻结 setup 的序列时间。[全部逐次数据](/Users/duoduo/Documents/温州肯恩大学/科研/Nonlinear Poisson/FD_LGF/audit_20261105/evidence/amg_scan.jsonl)、[含重复波动的汇总](/Users/duoduo/Documents/温州肯恩大学/科研/Nonlinear Poisson/FD_LGF/audit_20261105/evidence/amg_summary.json)。

## 3. [P1：计时结论；P2：计时口径。确定] 原残差时间未重复，跨批次差异足以动摇 LU/Cholesky 推荐

定位：[H53 IR 表](/Users/duoduo/Documents/温州肯恩大学/科研/Nonlinear Poisson/FD_LGF/audit_20261105/evidence/original_snapshot/h53_IR.csv:1)、[H55 IR 表](/Users/duoduo/Documents/温州肯恩大学/科研/Nonlinear Poisson/FD_LGF/audit_20261105/evidence/original_snapshot/h55_IR.csv:1)、[H53 报告源文:84](/Users/duoduo/Documents/温州肯恩大学/科研/Nonlinear Poisson/FD_LGF/regions/report/report_btfd_lu.tex:84)。

| 同一方法 | H53 总时间 / s | H55 总时间 / s | 总内层迭代数 |
|---|---:|---:|---:|
| AMG3D | 34.667 | 29.226 | 37 / 37 |
| bt-fd | 22.943 | 18.325 | 16 / 16 |

同方法下降分别为 15.7% 与 20.1%，残差历史亦几乎相同。三遍中位数仅用于 R 自报测试；IR 每批一遍。两批各自都显示 bt-fd 更快，但其比值约为 0.662 和 0.627，不能把单个 0.66 当作稳定性能常数。

更具体地，报告用 H52 的 CHOLMOD 17.4 s 对 H53 LU 22.9 s，建议比赛用 Cholesky；另一批 LU 已是 18.3 s。现有证据没有分离分解器作用与运行批次作用。相同迭代数意味着这个推荐应先做同批次、交错顺序的重复比较。

计时口径另有遗漏：[run_ir.py:48](/Users/duoduo/Documents/温州肯恩大学/科研/Nonlinear Poisson/FD_LGF/regions/run_ir.py:48) 在计时前构造 DDResidual，ELL 数据构造成本不计入所谓总时间；该对象的两张表在 n=6,291,456、每行至多 9 项时约占 906 MB 十进制内存。计时中包含每次 DD 残差计算，但不包含这个专用准备成本。它对两种方法是公共成本，补计通常会把时间比向 1 拉近，并不自动逆转胜负。

建议：同时报告公共准备、方法 setup、solve、IR 残差评估和全过程时间；每方法独立进程，随机或 ABBA 次序，预热加重复，报告分布，保留 RAM/压缩/换页增量。在有效原残差达标条件下再决定 LU/Cholesky。

## 4. [P1：G8 时间不可作干净基准；原因尚不确定] H56 存在剧烈计时异常与共同驻留问题

定位：[seq_lock.py:40](/Users/duoduo/Documents/温州肯恩大学/科研/Nonlinear Poisson/FD_LGF/regions/seq_lock.py:40)、[G8 日志](/Users/duoduo/Documents/温州肯恩大学/科研/Nonlinear Poisson/FD_LGF/audit_20261105/evidence/original_snapshot/h56_128_10_2_1_0_2e-2_1.log:1)。两个 KSP/预条件子同时保存在 ksps 中，bt-fd 永远先运行；两个方法的矩阵、Krylov 工作区和预条件子共同驻留。这会使内存紧张时的测量依赖另一方法的存在。

G8 的 bt-fd 始终 18 次、AMG 始终 20 次；bt-fd 前七步 solve 约 14.5–17.4 s，后三步 63.0、83.5、54.1 s。AMG 同时从约 11 s 升到 55.8、31.7、21.7 s。这是明显运行条件变化，不能解释为算法迭代恶化。日志没有同步 RSS/换页数据，交换、竞争、压缩或降频的具体贡献无法追溯。

按未删改的记录，G8 总时间比为 1.699。它必须保留并标为受干扰的测量；不能只剔除后三步、留下看起来更快的比值。建议各方法在独立进程运行完整同一序列，按统一规则拒收资源异常的整轮后配对重跑。seq_lock 把 PETSc 矩阵构造放在方法计时之前，也应把公共构造单列。

快照中 G1–G10 已完成，G11 未完成。G1–G6 时间比分别 0.422、0.375、0.518、0.720、0.545、0.448；G7–G10 为 0.634、1.699、1.126、2.284。故协议第 3 条“至少 3/4 获胜”按当前原始时间只有 1/4，未成立；第 2 条成立，第 6 条中位比约 0.590。完整的 220 次成功与 G11 步数稳定性在该快照仍不能验收。这个判断保留受干扰结果，不把它误当作纯算法反证。

## 5. [P2，确定，小规模复现] 合成组装改变了标称不变的反应项

定位：[gen3t.diffusion:40](/Users/duoduo/Documents/温州肯恩大学/科研/Nonlinear Poisson/FD_LGF/solver_challenge/gen3t.py:40)、[gen_gen.py:70](/Users/duoduo/Documents/温州肯恩大学/科研/Nonlinear Poisson/FD_LGF/regions/gen_gen.py:70)。面通量成对存储，但对角线用双精度逐面累加，之后再加较小的反应项；1e17 量级的相消使“扩散行和为零”的代数守恒性质在存储矩阵中失真。

独立测试 gen_gen(m=32,n=1,delta=0.1)，用 math.fsum 对已存储离子块逐行求和：热核标称 C3+W 约为 512。phase=0 时实际为 472–552；phase=1.35 时为 464–552，偏差分别可达 7.81% 和 9.38%。这是小规模同生成器复现，不声称这些区间就是 m=128 全套数据的区间。[结果](/Users/duoduo/Documents/温州肯恩大学/科研/Nonlinear Poisson/FD_LGF/audit_20261105/evidence/ir_exact.json)。

所以时间步不只体现指定的导热系数扰动，还带有组装舍入引起的小反应模态扰动。该问题不说明求解存储矩阵的结果错误，但会削弱“来自特定 PDE/只改变 k(T)”的解释。

修复建议：先输出每层的高精度行和与目标反应项偏差。若需要保真物理序列，将扩散以成对通量和反应项分开表示/计算，或采用足够精度的系数存储与运算。仅在组装阶段提高累加精度、最终又把巨大对角和小反应压成一个 float64，并不能完全保留小反应项。

## 6. [P2，确定缺陷；未证明影响现有 H55 数字] FDMBox 的一格厚方向忽略边界

定位：[rsolve.py:208](/Users/duoduo/Documents/温州肯恩大学/科研/Nonlinear Poisson/FD_LGF/regions/rsolve.py:208)。当某轴长度 n_=1，直接设特征值为 0，绕过两端 Dirichlet 和远端 Robin 贡献。内部单单元盒、kappa=10、r=1、dirichlet=True 的逆应为 1/(6*10+1)=1/61，实际返回 1，差 61 倍。[复现](/Users/duoduo/Documents/温州肯恩大学/科研/Nonlinear Poisson/FD_LGF/audit_20261105/evidence/numeric.json)。

修复：长度 1 时也分别累加两端边界的标量对角贡献（内部 Dirichlet 每端 kappa，外远面 Robin 为 beta），再作一维特征分解。补一格厚片、线和单点盒用例。当前所测 H55 同心球粗区域是否出现这样的包围盒尚未证实；不能用这个缺陷否定其已有时间表。

## 7. [P2，确定的推断范围问题] H56 的“稳定”只覆盖受限的扰动和右端项

定位：[gen_gen.py:58](/Users/duoduo/Documents/温州肯恩大学/科研/Nonlinear Poisson/FD_LGF/regions/gen_gen.py:58)、[seq_lock.py:26](/Users/duoduo/Documents/温州肯恩大学/科研/Nonlinear Poisson/FD_LGF/regions/seq_lock.py:26)、[报告 22 结论源文:84](/Users/duoduo/Documents/温州肯恩大学/科研/Nonlinear Poisson/FD_LGF/regions/report/report_gen.tex:84)。

除组装舍入外，跨步只让 k1、k3 乘同一个有界平滑场，热容、交换、反馈和电子导热参数均不随步数变化；所有锁定用例固定 F=2e-11、G=2e-4。右端项是同一个平滑 old 场乘新的 1% 噪声，三个场共享该 old 场；x0 始终为零。这样保留了块消去最有利的弱反馈、弱电子扩散结构，时间变化主要检验同一邻域内的预条件子复用。

delta=0.02 时，任意两相位的同一点导热系数比最多为 1.02/0.98=1.04082；delta=0.1 时最多为 1.1/0.9=1.22223。漂移不会随步数无界累积。因此把相位序列跑得更久，也不能直接推出实际过程何时必须重建。

另一个容易忽略的几何限制：n=1 分支中心固定在 (0,0,0)，seed 不改变几何，即单胶囊始终接触三个对称边界；G1–G6 没有测试同等体积的内部单胶囊。建议添加内部平移、容量/交换/电子扩散的独立变化、保持物理一致的右端项与双方共同的前一步初值，再判断复用寿命。报告用“在实际的时间推进计算里”超出了现有证据；更准确的说法是“在所构造的有界系数扰动序列上”。

## 8. [P2，确定] 不能把不同口径的特征比例拼成约 16% 的胜负阈值

H53 表中的 7.7%/9.2% 为外扩前 core_fraction：[表注](/Users/duoduo/Documents/温州肯恩大学/科研/Nonlinear Poisson/FD_LGF/regions/report/report_btfd_lu.tex:70)。H56 feature 列是两个块外扩后的最大 len(F)/N：[seq_lock.py:58](/Users/duoduo/Documents/温州肯恩大学/科研/Nonlinear Poisson/FD_LGF/regions/seq_lock.py:58)。两者并非同一个量；H53 比单次 setup+solve，H56 比十步摊销后的总时间。

快照中 16.28% 的 G9 输了，但 9.14% 的 G4 赢了；这只给出有限采样，不是 16% 的临界定理或已验证分类规则。稀疏 LU 成本还取决于连通性、填充、排序与复用次数。建议记录每块 core_fraction、dilated_fraction、因子非零数/内存和 setup；任何选择 bt-fd/bt-fdd 的阈值均须另设未参与拟合的验证集。

## 9. [P2，确定] 历史锁定文件和结果的复现链不完整

协议文件的 SHA256 均匹配；当前 H56 清单内文件也全部匹配。当前 H53 清单中的 rsolve.py、btfd.py、run_real.py、run_ir.py 已不匹配，H55 的 rsolve.py、run_real.py 不匹配。后续报告说明过方法扩展，这并不自动意味着违规改动，但当前工作树不能直接视作 H53/H55 锁定源码。

此外清单未覆盖 bt_elim.py、ir_fp128.py、fdlgf_real.py、gen3t.py 等实际依赖；这些文件可以改变 AMG、DD 残差、Schur 补和合成矩阵，且不触发该清单核验。顶层 Git 中 regions 当前也是未跟踪目录，没有从该历史中取得 H53/H55 对应提交。

建议保存每次实验完整可达源码的归档/提交、环境版本与有效选项、输入矩阵/右端项散列、最终 x_hi/x_lo 和结果。run_ir 当前 CSV 没有最终残差值、内层 reason、解向量；虽日志有残差，仍不足以离线用另一检查器重新验收。脚本也应对未知方法名报错，避免拼写错误落入 bt-fd 分支。

## 独立数学与残差核验：通过了什么

### DD 残差及迭代修正

two_sum/two_prod 实现与低位解更新方向一致；DDResidual 的低阶累计不是任意动态范围下的严格精确算术，但在这组数据尺度下没有发现导致错误达标的缺陷。对实际生产 inner()，m=12 三温系统最终解全部 5,184 行用 Fraction 精确重算：

| 方法 | DD 相对残差 | 精确有理数相对残差 | DD 与精确残差之差 / norm(b) |
|---|---:|---:|---:|
| AMG3D | 6.425371694e-12 | 6.425371700e-12 | 3.465e-20 |
| bt-fd | 2.597859139e-9 | 2.597859139e-9 | 3.400e-20 |

两种方法均通过 1e-8；各内层 reason 为 2。把最终解舍入成单 float64 后，残差分别回到 1.524e-4、1.842e-4。这验证了需要保留 hi+lo 的实质原因，也说明必须保存最终高精度表示才能复核。

真实矩阵 H53/H55 日志最后 AMG 为 4.53e-10，bt-fd 为 5.61e-11。这是历史日志证据，本次没有重新求解完整真实矩阵。共同的 DD 验收对两方法在阈值层面一致。run_ir 不检查内层 reason 是诊断缺项；只要外层独立真实残差确实达标，内层未达到指定容差也不自动使外层解无效，应避免误报成必然的假收敛。

本次验证的是所存储 A,b 上的原残差公式。Python 输出没有经过比赛官方可执行检查器的实测。本地 C 头文件支持 SC26_SOLUTION_FP128，但上游 Gitee 页面本次无法读取；官方接收路径与规则合规性不在本次已确认范围。建议区分“采用官方残差定义”和“已通过官方程序验收”。

### 平衡形式、Schur 补和块三角

ScalarFDLGF 中，令 Q 仅在 F 上解 M_FF，有 w_F=0、w_C=r_C-M_CF M_FF^{-1}r_F；输出 z_C 与 M_FF^{-1}(r_F-M_FC z_C)。在特征区求解精确的代数假设下，确实等于 Q+(I-QM)G(I-MQ)。本次以独立稀疏直接解构造参考，m=5 中等条件矩阵相对差为 8.97e-17。

DeflatedFDLGF 以 W=[I_F,V] 独立构造 W^T M W，并与生产结果比较，m=12、一个非平凡粗区域的相对差为 3.14e-16。这里 G 在 F 上为零，与代码掩码匹配。测试证明这两个具体样例的代数一致性，不证明高对比度下所有浮点等式都精确。

schur_ion 使用 a23=a32=-c 的假设，将 a33-c²/d22 重写成 (a33-c)+c(d22-c)/d22，代数正确。BTFD 对应删除 A21、用 diag(A22) 替代 A22 后的约化块系统；它是原 A 的预条件子，而非原系统的精确电子 Schur 消去。外层仍作用原 A，因此这些近似本身不是方程被偷偷换掉的证据。中等条件的显式约化块系统与代码解相对差为 2.61e-16。

在高对比度小型生成矩阵上，两种双精度直接算法的解仍相差约 3.29e-5，虽然缩放残差约 3.49e-17；这说明不能把另一个双精度稀疏 LU 当成该问题的精确解基准。现有全局、单随机向量对的 linearity defect 也不等价于局部求逆精度；建议补分场和低能模态检查。

## 建议处理顺序

1. 让 AMG 参数真正可覆盖并记录有效配置；统一主验收为高精度原残差，同时保存最终 hi/lo。
2. 隔离进程、记录资源状态，对真实矩阵做 AMG 参数与 LU/Cholesky 的同批次配对重复；补计 DD 初始化。
3. 修复 FDMBox 单层方向边界；记录/控制合成矩阵的反应项组装误差。
4. 将报告中的稳定加速、实际时间推进、16% 阈值和提交分解器选择收窄为当前证据支持的表述。

## 复现与材料

所有新增脚本仅导入/读取生产代码，不回写它们。推荐在空闲机器运行，避免与原 H56 作业共同计时。

```sh
cd '/Users/duoduo/Documents/温州肯恩大学/科研/Nonlinear Poisson/FD_LGF'
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 /opt/anaconda3/envs/hypre_env/bin/python audit_20261105/check_numeric.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 /opt/anaconda3/envs/hypre_env/bin/python audit_20261105/check_ir.py
/opt/anaconda3/envs/hypre_env/bin/python audit_20261105/scan_driver.py
/opt/anaconda3/envs/hypre_env/bin/python audit_20261105/summarize.py
```

环境：NumPy 2.2.6、SciPy 1.15.2、PETSc 3.25.6；hypre/MUMPS 可用。OpenBLAS 0.3.34 编译为 USE_OPENMP，本次直接查询 OMP_NUM_THREADS=1 时实际线程数为 1，未发现“仅设置 OMP 导致本机 BLAS 偷跑多线程”的证据。

原始快照与散列见 [manifest](/Users/duoduo/Documents/温州肯恩大学/科研/Nonlinear Poisson/FD_LGF/audit_20261105/evidence/original_snapshot/manifest.json)。项目研究日志实际位于相邻 axy_lgf/autoresearch_lgf_amg，已只读核查并在 evidence/original_snapshot 留存对应文本。H56 原始结果在审计期间继续更新，本报告的判定固定于上述快照，避免把半途中结果当成最终结果。

审阅 PDF 原文：[报告19](/Users/duoduo/Documents/温州肯恩大学/科研/Nonlinear Poisson/FD_LGF/总结/19_11月5日_H53_bt-fd改用LU重新锁定验证.pdf)、[报告21](/Users/duoduo/Documents/温州肯恩大学/科研/Nonlinear Poisson/FD_LGF/总结/21_11月5日_H55孤岛粗空间bt-fdd_锁定实验.pdf)、[报告22](/Users/duoduo/Documents/温州肯恩大学/科研/Nonlinear Poisson/FD_LGF/总结/22_11月5日_H56探索_泛化测试与跨时间步复用.pdf)。已提取全部页面文本，并目视核查关键图表/结论页；报告22明确是探索阶段五步实验，应与后来锁定的十步 H56 结果区分。
