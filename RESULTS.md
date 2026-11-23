# 方法二对比结果（H31 二维 / H32 三维）

完整报告（含全部表格、图、文献对比）：`report/report.pdf`。原始数据：`results/benchmark.csv`、`results/traces.json`（二维），
`three_d/results/benchmark3d.csv`、`three_d/results/traces3d.json`（三维）。计时为单线程，同一台 8 核 Mac。

## 二维（n = 127–1023，χ = 1e2–1e8，方案 38eab086）

| 算例 | 迭代：方法二 / BoomerAMG / GMG / 传统 LGF | 误差到 1e-8 的迭代 | n=1023、χ=1e8 总时间 |
|---|---|---|---|
| A χ^x | **2–3** / 7–8 / 7–8 / 110–1000（χ≥1e4 不收敛） | 2–3 / 5 / 5–7 / 68–73 | **0.25 s** / 1.14 s / 0.90 s / 46 s |
| B 阶跃层 | **1–3** / 8–9 / 7 / 15–58 | 3 / 5–6 / 4–5 / 10–72 | **0.20 s** / 1.35 s / 0.94 s / 3.0 s |
| C χ^x+9 圆 | **7–8** / 9–11 / 48–95 / 不收敛 | 9–10 / 8–9 / 55–89 / – | 4.7 s（w=1/64: 2.3 s）/ **1.5 s** / 8.2 s / 39 s |
| D χ^x+20 随机圆 | 8–10 / 8–11 / 73–114 / 不收敛 | 10–11 / 7–11 / 75–108 / – | 5.0 s（w=1/64: 2.7 s）/ **1.6 s** / 9.3 s / 38 s |

- A 上所有收敛方法的总误差前 3 位相同（n=1023、χ=1e8：4.7e-5）；传统 LGF 未收敛，χ≥1e6 时误差大 1.1–950 倍。
- 按残差停止时，方法二的代数误差（C、D：1e-8–7e-6；B χ=1e8：3.8e-4，只迭代 1 次）比 BoomerAMG（≤1.5e-8）大，
  原因是低系数区的误差在残差里几乎看不到；要按误差计数或收紧容差。
- 方法二的迭代数与 χ、n 无关；时间在有夹杂时由 LU 决定（w=1/16 时 LU 区域占网格 48–57%）。
- 预测：1、3、6 成立；2 部分成立（LGF 误差大 ≥10 倍只在 n≥511 成立）；4 对 BoomerAMG 成立、对 GMG 不成立（GMG 改 ω=4/5 后更快）；
  5 前半成立、后半不成立（最快环宽仍是 BoomerAMG 的 1.44–1.82 倍）。

## 偏离预注册
- GMG：原方案用项目 `axy_lgf` 的 GMG（Laplace 上 ω=1/2）。核查发现比教科书 ω=4/5 弱（V(2,2) 收敛因子 0.30 vs ≈0.13），
  经确认改为 ω=1.6/行和（Laplace 上 = 4/5），二维 GMG 全部 64 组重跑；旧结果在 `results/benchmark_gmg_omega05.csv`。

## 三维（n = 31–127，最多 2×10⁶ 未知量，χ = 1e2–1e8，方案 06cc1590）

| 算例 | 迭代：方法二 / BoomerAMG / GMG / 传统 LGF | 误差到 1e-8 的迭代 | n=127 总时间（各 χ） |
|---|---|---|---|
| A χ^x | **2–3** / 5–9 / 8–9 / 68–500（χ≥1e4 基本不收敛） | 2–3 / 5–8 / 6–8 / 54–244 | **0.34–0.44 s** / 7.1–8.9 s / 2.2–2.9 s / 10–48 s |
| B 阶跃层 | **1–3** / 7–11 / 9 / 19–62 | 3 / 6–9 / 5–6 / 12–87 | **0.23–0.47 s** / 12–13.5 s / 2.2 s / 1.9–5.8 s |
| C χ^x+27 球 | **10–12**（w=1/32）/ 11–15 / 57–103 / 不收敛 | 10–14 / 10–14 / 64–100 / – | 9.2–9.9 s（w=1/64）≈ BoomerAMG 9.3–10.1 s |
| D χ^x+30 随机球 | **11–13**（w=1/32）/ 13–17 / 70–135 / 不收敛 | 11–15 / 12–16 / 80–129 / – | **5.5–6.0 s**（w=1/64）vs BoomerAMG 11.0–11.7 s |

（时间列为各 χ 的范围。）
- 三维的夹杂是分开的小球，LU 区域只占网格 6–15%（二维 w=1/16 时 48–57%），所以方法二在三维有夹杂时与 BoomerAMG 持平（C）或快约 2 倍（D）。
- BoomerAMG 在三维迭代数随 n 增长（11→17）、setup 3.4–6.2 s；GMG（ω=6/7）在 A、B 上比 BoomerAMG 快 3–6 倍。
- w=1/32 迭代最少，但三维 LU 填充增长快（C：setup 0.24 s → 17 s，n=63 → 127），w=1/64 总时间更短。
- 精度：A 上收敛方法总误差相同（n=127、χ=1e8：2.53e-3）；按残差停止时方法二在 C、D 的代数误差 1.4e-9–4.5e-7，BoomerAMG ≤1.5e-8。
- 预测 1、2、4 成立；3 成立但 D 上方法二明显更快（预测是“不分胜负”）；5 基本成立（46/48 和 48/48）。

## 报告
`report/report.pdf`（副本：`方法二_二维三维对比报告.pdf`），源文件 `report/report.tex`，表格由 `report/make_tex.py` 从 CSV 生成，
图由 `make_figures.py`（二维）和 `make_figures.py 3d`（三维）生成。

## H33：不只依赖 x 的高对比度系数（无传统 LGF；“方法二”改称 FD-LGF）
完整报告：`high_contrast/report_hc/report_hc.pdf`（副本 `FD-LGF_高对比度系数与BoomerAMG弱点.pdf`）。数据：`high_contrast/results/`
（`scan.csv` 二维探索 n=255/511，`scan3d.csv` 三维探索 n≤127，`hc1023.csv` 二维确认 n=1023，方案 c2ff7672）。
- BoomerAMG：二维 10 类、三维 5 类、χ≤1e8 全部 7–13 次，**没有找到迭代失败的例子**；三维 n=127 时 8–17 s（setup 5–8 s），比 GMG 慢 2–5 倍。
- GMG：细斜条纹最多 368 次、随机两相 106 次。
- FD-LGF：可分离背景最快（三维薄层 0.3–0.5 s，比 BoomerAMG 快 25–40 倍；三维管道网约快 2 倍；二维 sep2 3–4 次，但 n=1023 时稠密 FDM 使时间与 BoomerAMG 持平）；
  通道、绝缘墙为 BoomerAMG 的 1.1–2.1 倍；不可分离系数（棋盘、随机场、斜条纹、随机两相）局部区域 46–100%，退化为直接 LU，慢 1.8–14 倍；三维绝缘板 115 s 以上。
- 预测：1、2 成立；3 迭代成立、时间不成立；4 部分成立；5 walls 成立、chan 勉强不成立（2.09 倍）。

## H34 — rotated / field-aligned anisotropic tensor K (`anisotropic/`, report `anisotropic/report_aniso/report_aniso.pdf`)
- 9-point SPD stencil, not an M-matrix. n = 1023, χ ∈ {1e2, 1e4, 1e6}, 6 families. Protocol bf86f18d.
- BoomerAMG degrades with n on circ (144–241 its at χ ≥ 1e4), disk (98–218), band (56–65); constant rotation ≤ 33.
- FD-LGF (isotropic background + LU on the anisotropic region): disk/band 4–8 its, flat in n; 0.29–0.31× BoomerAMG
  time on band, 0.53–0.66× on disk (0.36–0.48× with w = 1/64). Global anisotropy: 95–1000 its, 1.8–8× slower.
- Predictions 1, 2, 4, 5 supported; 3 supported for k8 and band time, refuted for disk time.

## H35/H36 — SolverChallenge26_01 type 3T radiation-diffusion matrix (`solver_challenge/`, report `solver_challenge/report_sc/report_sc.pdf`)
- Real matrix not available; surrogate with the exact sparsity (order 6,291,456, nnz 52,133,888), SPD M-matrix.
- Plain FD-LGF fails (98–99 % of cells are features; background only: no convergence; scalar Green–Jacobi 1182–1801 its).
- FD-LGF-3T (`fdlgf3t.py`): 3×3 point-block Green–Jacobi LGF + block-Jacobi smoothing + box-constant coarse space,
  multilevel W-cycle, balanced form. Ablation m = 32: 1663 → 112 → 50 → 27 its. m = 128 (protocol eb34886e):
  37 / 53 its to 1e-8 (dt = 1e-3 / 10), 26.2 / 33.8 s, peak RSS < 5.7 GB; BoomerAMG 3D settings 14.0 / 17.3 s
  (1.35–1.95× faster), default BoomerAMG 19.6 / 40.8 s. All 3 predictions supported.

## H37/H37b — speed-up (`solver_challenge/fdlgf3t.py`, report section 7)
- LGF (Green function) part shown useless on the 3T class; balanced-form variants stay 1.3–2.0× slower than BoomerAMG.
- agg3t-w (3×3 block-Jacobi V(1,1), box-2 aggregation, W-cycle; no LGF): m = 128 0.41–0.58× BoomerAMG-3D time,
  25 / 36 its to 1e-8, ≤ 4.3 GB. Predictions H37: 1–3 refuted, 4 supported; H37b: 4/4 supported.

## H38 — nonsymmetric 3T surrogates (report section 8)
- emis: agg3t-w 0.39–0.57× BoomerAMG-3D time; BoomerAMG stalls at 1e-8 for dt = 10. scaled: agg3t-w converges for
  both dt; BoomerAMG fails (plain) or is 1.8–3.4× slower / fails at dt = 10 (with diagonal scaling). 2/2 predictions.
