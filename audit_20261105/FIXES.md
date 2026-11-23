# 审计处理记录（2026-11-05）

对应 [AUDIT.md](AUDIT.md)。AUDIT.md 和 evidence/ 未改动。说明与数字见报告 24：`总结/24_11月5日_审计修复与真实矩阵配对复测.pdf`。

| 审计项 | 状态 | 处理 |
|---|---|---|
| 1 自报阈值≠同精度 | 代码已处理；报告 19、21–23 开头已加更正说明 | `run_ir.py` 记录 DD 原残差、缩放残差、内层 reason，保存 hi/lo；`run_real.py` 文档注明自报残差仅作诊断 |
| 2 AMG 选项被覆盖 | 已修 | `bt_elim.amg_defaults()` 只写缺失的选项；`PETSC_OPTIONS` 可注入；`provenance.py` 保存 ksp.view、选项库、版本、输入散列 |
| 3 单遍计时 / LU 与 Cholesky | 已复测 | `remeasure_real.py`：独立进程、每轮随机顺序、5 轮；`prep_s/solve_s/resid_s` 分列 |
| 4 H56 共同驻留 / G8 异常 | 已重跑 G8–G11 | `seq_isolated.py`：时间比 1.67、1.09、1.50、6.71（H56 原值 1.70、1.13、2.28、4.39）；一遍，G1–G7 未重跑；G11 的 bt-fd 自身换页 |
| 5 反应项舍入 | 只做了诊断 | `diag_reaction.py`，m=128 已测；未改生成器，改动会使已有生成矩阵全部变化 |
| 6 FDMBox 单层方向 | 已修 | 两端面分别累加；真实矩阵 bt-fdd 的 4 个盒子都是 41–51 格厚，H55 真实矩阵结果不受影响 |
| 7 H56 推断范围 | 报告 22、23 开头和报告 24 已收窄 | 内部单胶囊、物理一致右端项等新用例未加 |
| 8 约 16% 阈值 | 报告 23 开头和报告 24 已收窄 | `run_ir` 现分块记录 core/dilated 比例和因子非零元 |
| 9 复现链 | H56 已恢复；H50、H52、H53、H55 无法恢复 | 见下；`lock_manifest.py` 沿 import 生成全依赖清单；每次运行留存配置和散列 |

## 锁定文件

`regions/`、`multifield/` 此前不在 Git 里，现已入库。标签 `h56-lock-sources` 是 H56 锁定状态，取出后 `protocol_h56.sha256` 全部通过：

```sh
git archive h56-lock-sources FD_LGF/regions | tar -x -C /某个空目录
```

当前工作区各清单的状态：

| 清单 | 不匹配的文件 | 能否恢复 |
|---|---|---|
| H56 `protocol_h56.sha256` | `rsolve.py`、`seq_lock.py`（审计修复） | 能，用上述标签 |
| H55 `protocol_h55.sha256` | `rsolve.py`、`run_real.py`、`run_ir.py` | 不能，审计前已不匹配，且没有历史版本 |
| H53 `protocol_btfd_lu.sha256` | `rsolve.py`、`btfd.py`、`run_real.py`、`run_ir.py` | 不能，同上 |
| H52 `protocol_btfd.sha256` | `btfd.py`、`rsolve.py`、`run_real.py`、`run_ir.py` | 不能，同上 |
| H50 `protocol_amfr.sha256` | `amfr.py`、`run_mf.py` | 不能，同上 |

这些清单有意不重新生成，否则就失去了校验的意义。H50–H55 的结果仍在 `results/`（本地，不入库）和 `evidence/original_snapshot/`，但无法从源码逐字节复现。

下次锁定：先提交，再运行 `lock_manifest.py`，它会对未提交的文件发出警告。

`rsolve.py` 的改动只涉及单层盒子和默认关闭的 `CHOLESKY`，H56 走的 `ScalarFDLGF` + LU 路径不变。不预设选项时，`seq_lock.py` 的行为也不变。

## AMG 选项覆盖的范围

已改：`regions/run_ir.py`、`run_real.py`、`seq_lock.py`、`seq.py`，`solver_challenge/bt_elim.py`、`run_gen.py`、`ir_fp128.py`、`run_sc.py`，`multifield/amfr.py`、`run_mf.py`。

没改：`generalize/diag.py`、`generalize/snn/*`、`three_d/solvers3d.py`。它们的参数由调用方显式传入，或者每次用新前缀，这是设计如此，不属于这个缺陷。

## 旧报告

报告 19、21、22、23 的 LaTeX 只在开头增加了更正说明框，正文未改（`git diff` 只有新增行）。PDF 不入库（`.gitignore` 的 `*.pdf`），未加说明的原始 PDF 保存在本地 `总结/原始版本_审计更正前/`。

## 验证

```sh
cd regions
OMP_NUM_THREADS=1 /opt/anaconda3/envs/hypre_env/bin/python test_audit_fixes.py
```

旧版 `FDMBox` 用同一组用例检查：12 个组合中 7 个失败，n≥2 的盒子通过。原审计脚本 `check_numeric.py`、`check_ir.py` 会覆盖 evidence/，因此没有重跑。

逐脚本用 `-*_ksp_view` 确认 `PETSC_OPTIONS` 设的阈值生效（`seq_lock`、`seq`、`run_gen`、`ir_fp128`、`amfr.vcycle`）。`run_sc.py` 只改了调用并通过编译，没有实际运行。

默认行为与标签版本对比：`run_gen`（`m=16`，80 行）和 `run_mf`（2 个算例）迭代数全部相同。`run_gen` 的残差有差异，但同一版本连跑两遍差异同样大（缩放残差相对差最多 2.7e-3），说明这些求解器不是逐位可复现的；核对历史结果时看迭代数和数量级，不要求残差逐位相同。

复测原始数据另存于 `rerun_evidence/`。
