"""Generate a final report only after the complete audit, preserving negative results."""
import json,statistics
from pathlib import Path
H=Path(__file__).resolve().parent
a=json.loads((H/'aggregate.json').read_text());d=json.loads((H/'summary.json').read_text())
audit=json.loads((H/'AUDIT.json').read_text())
assert a['records']==200 and audit['status']=='passed'
diags=[json.loads(x) for x in (H/'diagnostics.jsonl').read_text().splitlines()]
passed=a['lifting_vs_direct_criterion_pass']
lines=['# Boomer 补偿研究：第三轮强对照与机理检查','',
       '完成日期：2026-11-17（北京时间）。2026-11-16锁定方案；前两轮文件及算法保持不变，本轮测试8个新参数案例。',
       '',
       '**判断：实用性与研究贡献需要分别评价。** '+
       ('本轮稀疏提升通过预设的直接谱补偿比较门槛，但仍不能据此保证首创或通用最优。' if passed else
        '本轮稀疏提升没有通过预设的直接谱补偿比较门槛。应保留它作为有效的工程候选，但下调“相对已有谱补偿有明确优势”的判断。'),
       '',
       f"固定的第二轮自适应方案相对逐例较快的两套 Boomer，在 **{a['adaptive_ratio_wins']}/8** 个案例上更快；中位总时间比为 **{a['adaptive_ratio_median']:.3f}**（节省{100*(1-a['adaptive_ratio_median']):.1f}%），CPU时间比中位数为{a['adaptive_cpu_ratio_median']:.3f}。",
       f"但等96步Arnoldi预算下，稀疏提升相对逐例最快的直接2/16/32模态补偿，只在 **{a['weighted_direct_ratio_wins']}/8** 例更快，中位时间比 **{a['weighted_direct_ratio_median']:.3f}**；CPU比中位数{a['weighted_direct_cpu_ratio_median']:.3f}。两项预设门槛分别为：实用性 **{'通过' if a['practical_criterion_pass'] else '未通过'}**；提升对直接谱补偿 **{'通过' if passed else '未通过'}**。这都是有限重复的点估计。",
       '',
       '## 本轮做了什么','',
       '10种方法：Boomer Pmax=4/8、原封不动的第二轮自适应方案、固定96步的加权值域帽函数、非加权帽函数、几何局部POU补偿、直接保留2/16/32个Ritz模态，以及PyAMG 5.3.0实际的αSA实现。没有把手写的简化局部基冒称αSA。',
       '',
       '比较采用新参数C=[.21,.73,.35]和D=[.47,.19,.81]。511²覆盖circ/disk/ellipse/two_vortices，χ=10⁵，新平滑右端项；1023²覆盖circ/disk/two_vortices的真正随机forcing，以及χ=10⁴的circ平滑右端项。χ主要表示张量各向异性比，不能与SC26_01的整体对比度直接等同。每个小案例3次重复，每个大案例2次，单线程、随机运行顺序。具体身份见LOCK.json。',
       '',
       f"共 **200条计划任务记录**，其中{a['completed_solver_runs']}次求解返回了独立残差审计，{a['successes']}条成功，{a['failures']}条不成功；超时{a['timeouts']}条，因同案例/方法已超时而跳过后续重复{a['budget_skipped']}条。成功记录最大double-double真实相对残差为 **{a['max_successful_residual']:.3e}**。所有非αSA记录均成功；失败不计入加速排名。另有4个小规模机理案例、40次补偿求解，独立于200条正式记录。",
       f"有制造解的成功记录中，最大相对解误差为 **{a['max_successful_manufactured_error']:.3e}**；它与相对残差是两个指标，不能把残差容差当作解误差保证。",
       '',
       '## 总时间结果','',
       '时间均含全部预条件器、坐标、基函数构建及求解，不含矩阵装配和最终DD审计。下表每格为秒；直接谱列允许逐例选择三个秩中最快者，是有利于该对照的选择。',
       '',
       '| 案例 | 较快Boomer | 自适应提升 | 固定96步提升 | 最快直接谱 | 直接谱秩 |',
       '|---|---:|---:|---:|---:|---:|']
for r in a['per_case']:
    x=d[r['case']]
    lines.append(f"| {r['case']} | {x[r['baseline']]['total_s']:.3f} | {x['adaptive']['total_s']:.3f} | {x['weighted']['total_s']:.3f} | {x[r['best_direct']]['total_s']:.3f} | {r['best_direct'].replace('direct','')} |")
lines+=['',
        f"加权提升相对几何局部POU的中位时间比为 **{a['weighted_local_ratio_median']:.3f}**，在{a['weighted_local_ratio_wins']}/8例更快。相对非加权帽函数则为 **{a['weighted_hats_ratio_median']:.3f}**，在{a['weighted_hats_ratio_wins']}/8例更快。若非加权帽函数表现相近或更好，就不能把乘以坐标v本身包装为已经证实的主要性能贡献。",
        '',
        '所有10种方法、迭代数及逐例比值见RESULTS_TABLE.md；重复范围、CPU时间、内存峰值及空间维数见summary.json，原始记录见results.jsonl。正式图见comparison.png。',
        '',
        '## 机理诊断：不是只看迭代数','',
        '在255²上独立构造16个细网格Ritz探针，检查不同粗空间的A能量投影误差。它们是近似Ritz向量，不是精确特征向量；坐标来自63²网格上的96步构建，细网格探针来自128步构建。诊断计时不混入正式性能结果。下表取第3–16个探针的平均相对能量投影误差，越低越好。',
        '',
        '| 场景 | 直接2 | 直接16 | 直接32 | 稀疏提升 | 几何POU | 最大细网格Ritz相对残差 |',
        '|---|---:|---:|---:|---:|---:|---:|']
for x in diags:
    vals={r['tag']:statistics.mean(r['projection_error'][2:]) for r in x['comparisons']}
    lines.append('| '+x['fam']+' | '+' | '.join(f'{vals[t]:.3f}' for t in ['direct2','direct16','direct32','weighted','local'])+f" | {max(x['fine_probe_ritz']['ritz_residual']):.2e} |")
lines+=['',
        '理论上，完整帽函数的分片统一性使Z包含原始v及逐点v²。因此，理想提升空间对任意探针的最佳能量投影不会差于仅用同两个坐标的线性空间；代码中的舍弃阈值使这一性质需数值核查。本轮核查通过。但这不能保证比直接16/32模态更好，更不保证总时间或网格无关收敛。',
        '',
        '### 坐标旋转敏感性','',
        '将相同的两个坐标旋转，原始线性空间不变。直接谱补偿的不变性已数值验证；稀疏非线性提升一般会改变。下表为相同右端项的外层PCG迭代数；0°和90°应只是坐标置换/符号变化，核查一致。',
        '',
        '| 场景 | 0° | 22.5° | 45° | 90° |',
        '|---|---:|---:|---:|---:|']
for x in diags:
    lines.append('| '+x['fam']+' | '+' | '.join(str(r['iterations']) for r in x['rotations'])+' |')
lines+=['','### 单独探索的联合坐标原型','',
        '根据上述结构分析，另外实现了联合(v1,v2)坐标上的Delaunay线性插值：保留坐标点云凸包与原点，最远点采样补充中心，重心插值后去掉原点基函数。共同尺度归一化避免人为区分两个坐标轴；每行最多3个非零。它不在200条正式计时里，复用了诊断坐标，不能从这里宣称端到端加速。详见JOINT_PILOT.md。',
        '',
        '| 场景 | 原提升0°迭代 | 联合0° | 联合22.5° | 联合45° | 联合90° | 联合0°维数 |',
        '|---|---:|---:|---:|---:|---:|---:|']
for x in diags:
    jr=x['joint_rotations']
    lines.append('| '+x['fam']+f" | {x['rotations'][0]['iterations']} | "+' | '.join(str(r['iterations']) if r['ok'] else 'FAIL' for r in jr)+f" | {jr[0].get('dim','FAIL')} |")
joint_good=[r for x in diags for r in x['joint_rotations'] if r['ok']]
joint_wins=sum(x['joint_rotations'][0]['ok'] and x['joint_rotations'][0]['iterations']<x['rotations'][0]['iterations'] for x in diags)
maxchange=max((r['relative_preconditioner_change'] for r in joint_good),default=float('nan'))
lines+=['',f"联合原型成功{len(joint_good)}/16次；0°时相对原提升迭代数更少的场景为 **{joint_wins}/4**。旋转后对同一随机向量施加预条件器，最大相对变化为 **{maxchange:.3e}**。旋转稳定与收敛更快是两件事；若迭代没有优势，不应仅因每行更稀疏就优先投入。",
        '目标粗维数98，与原诊断提升上限相同；若凸包顶点超过预算就保留它们，并如实报告更大的维数。节点碰撞、近一维点云、瘦三角形和坐标误差仍可能限制效果。该组合使用的采样、Delaunay及重心插值都是成熟技术，尚未建立文献首创性。']
lines+=['',
        '旋转试验主动混合了两个坐标，不能直接解释为实际求解时会发生同等幅度的不稳定；只有特征值聚集等情形才容易出现子空间稳定而单个向量方向不稳定。它揭示的是构造的结构性敏感性。',
        '',
        '在固定值域节点、间距Δ下，行映射F(t)=[t h_j(t)]满足粗略的Lipschitz上界1+√2 M/Δ（|t|≤M）。因此坐标误差可能被窄分段放大；从Ritz残差到提升空间误差还需要特征间隙、归一化/节点移动和基矩阵条件数控制。详见THEORY.md。这里给出的是初步推导与数值诊断，不是新的统一收敛定理。',
        '',
        '## 公平性与尚未解决的比较','',
        '- 本轮“直接谱补偿”专指粗网格Bc Ac的Ritz向量延拓后形成的空间，并非直接求细网格BA的精确慢特征空间。粗细网格的Boomer层次不同，慢模态也可能不同；投影诊断显示了这种失配。结果不能外推成优于细网格谱消去或所有扩散滤波方法。',
        '- 固定预算比较的所有变体从零初值开始，无门控，96步粗网格Arnoldi。返回2/16/32模态时，最后的向量形成和逐模态残差核查成本不同，所以这是等Arnoldi步数，不是精确等墙钟构建时间。直接谱用高效dense BLAS和Cholesky，避免用CSR存稠密向量惩罚它。',
        '- 提升拥有更多但稀疏的列；直接谱拥有2/16/32个稠密列，不能说是等秩比较。几何POU仅近似匹配维数，且没有局部谱问题、能量最小化或多层重建，不能代表最佳GenEO/局部自适应方法。',
        '- 自适应方案仍有16步探测、门控和重启，与固定96步无门控变体并非单因素对照。第二轮已有相同门控的自适应/固定预算比较；本轮该列用于工程效果复核。',
        '- 非加权帽函数同时改变了零附近帽函数的处理，不是只删除乘子v的纯单因素消融。',
        '- αSA使用预先固定、未经调优的默认风格配置。PyAMG模块首次导入也落入其计时区间，而PETSc模块导入在计时外；不能据此比较非常小的启动成本。这不解释当前数十秒级差距，但仍是实现差异。该对照不代表αSA方法族的最佳表现。',
        '- 2026年diffusion-filtered coarse spaces论文只获得摘要和章节片段，没有全文算法或代码。本轮直接Ritz对照不是该论文的精确复现；不能宣称已击败它。',
        '- 两/三次重复只能支持当前机器上的中位数点估计。没有GPU、MPI、三维、非结构网格或SC26_01验证。前后轮案例不同，不能用各轮总体中位数相减衡量进步。',
        '- 正式任务在第132条后执行会话中断，约11小时后续跑剩余身份；源码与参数没有改动。跨会话计时会混入机器状态差异，应结合下述锚点检查评估小幅胜负，不能视为严格的同一批次重复。',
        '',
        '## 研究判断与下一步','',
        '现有证据支持继续保留这条方向，但应把课题收窄为“少量谱坐标如何低成本扩展成有效的稀疏补偿空间”，而不是“首次自适应修复AMG”。慢模态发现、平衡两层补偿、Ritz停止、局部近零空间、基于光滑误差的聚合都有明确先例。性能优于一个配置与方法首创是两个独立问题。文献及精确链接见LITERATURE.md。',
        '',
        ('联合坐标原型显示了初步迭代收益，值得进入包含全部构建成本的新验证轮次；当前诊断还不能证明总时间优势。' if joint_wins>=3 else '联合坐标原型尚未显示普遍的迭代优势，暂不应升级为主方案。')+
        '下一步应厘清哪些慢误差能由少量坐标的值域区分，以及何时坐标碰撞或精度不足会限制效果，并与更强的经典自适应方法配置比较。仅继续调整帽函数数量或预算，难以建立有说服力的新贡献。如果结构改进仍不能稳定优于直接谱补偿，就应定位为特定问题的工程优化。',
        '',
        '## 复现与审计','',
        '入口及命令见README.md。LOCK.json固定算法/依赖哈希；AUDIT.json记录正式检查。所有前两轮源码保持原样。THEORY.md、diagnostics.py和统计/画图文件不参与锁定求解路径。']
if (H/'ANCHOR_SUMMARY.json').exists():
    anchors=json.loads((H/'ANCHOR_SUMMARY.json').read_text())
    lines+=['','## 跨会话计时锚点','',
            '正式实验完成后，对全部8个案例各补跑2次Pmax=8，共16次，仅用于检查漂移，不混入正式中位数。下表为新会话锚点中位数/首132条中的旧会话Pmax=8中位数。它不构成对所有方法的漂移校正。',
            '', '| 案例 | 旧会话样本 | 新锚点样本 | 墙钟比 | CPU比 | 迭代一致 |',
            '|---|---:|---:|---:|---:|---|']
    for r in anchors:
        if 'total_s_ratio' in r:
            lines.append(f"| {r['case']} | {r['original_session_samples']} | {r['new_samples']} | {r['total_s_ratio']:.3f} | {r['cpu_s_ratio']:.3f} | {r['iteration_identity']} |")
        else:lines.append(f"| {r['case']} | 0 | {r['new_samples']} | 无参照 | 无参照 | — |")
conf=json.loads((H/'CONFIRMATION_SUMMARY.json').read_text())
if conf['triggered']:
    lines+=['','### 同会话配对复核','',
            '至少一个锚点超出预设±10%范围，因而统一对全部8个案例补做稀疏提升与原表最快直接谱秩的配对复核，共32次求解。没有只重测不利案例，也没有把新结果替换进原表。这是已知案例上的计时重复验证，不是新的泛化测试。',
            f"配对复核中，稀疏提升在 **{conf['wins']}/8** 例更快，中位墙钟时间比 **{conf['median_ratio']:.3f}**、CPU时间比 **{conf['median_cpu_ratio']:.3f}**。",
            '', '| 案例 | 稀疏提升 秒 | 直接谱 秒 | 秩 | 时间比 | 两个配对比值 |',
            '|---|---:|---:|---:|---:|---|']
    for r in conf['per_case']:
        pairs='/'.join(f'{p:.3f}' for p in r['pair_ratios'])
        lines.append(f"| {r['case']} | {r['weighted_s']:.3f} | {r['direct_s']:.3f} | {r['direct'].replace('direct','')} | {r['ratio']:.3f} | {pairs} |")
else:lines+=['','锚点均在预设±10%范围内，因此没有触发额外配对复核。即使如此，少量重复中的几个百分点差异仍不构成统计显著性证据。']
(H/'REPORT.md').write_text('\n'.join(lines)+'\n')
print('Wrote REPORT.md')
