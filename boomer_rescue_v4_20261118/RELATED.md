# 文献对照

| 已发表的原理 | 与本轮方法的交集及可检验的区别 |
|---|---|
| [Brandt 等，Bootstrap AMG](https://doi.org/10.1137/090752973) | 使用迭代和松弛改进测试向量，再拟合插值。细网格修正测试向量有明确先例。本轮把每条修正向量当作标量坐标，在值域直接生成一组帽函数。 |
| [Hu、Lin、Zikatanov，基于路径覆盖的自适应 AMG](https://arxiv.org/abs/1806.07028) | 根据光滑误差的等值集寻找路径、形成聚合并重建层次。两种方法均利用等值结构，本轮补偿可跨空间上不连通的节点，保留原 Boomer 层次。 |
| [Owhadi、Zhang，Metric based up-scaling](https://arxiv.org/abs/math/0505223) | 利用调和坐标表达粗尺度近似。本轮坐标由预条件算子的近似慢模态产生，服务于一次线性求解中的外部补偿。 |
| [Brannick、Falgout，Compatible Relaxation and Coarsening](https://doi.org/10.1137/090772216) | 通过松弛估计粗空间质量并指导粗化。本轮用辅助网格预条件算子的 Ritz 值决定是否延续同一个子空间计算。 |
| [扩散滤波粗空间，2026](https://doi.org/10.1016/j.camwa.2026.09.024) | 出版商摘要已描述预条件器感知的扩散滤波、SVD以及平衡补偿。本项目当前的区别在于非线性值域展开和行稀疏性。仍缺该论文实现的直接数值对比。 |
| [Stathopoulos、Orginos，eigCG](https://doi.org/10.1137/080725532) | 从CG轨迹提取特征向量和做谱消去均已有研究。本轮基于 PCG 起步轨迹的消融检验右端项依赖坐标。 |
| [Olson、Schroder、Tuminaro，能量最小化插值](https://lukeo.cs.illinois.edu/files/2011_OlScTu_energymin.pdf) | 在指定稀疏模式与候选向量约束下优化插值能量。本轮负结果涉及仅约束区间权重与支撑的简化目标。 |
| [Chen、Sun、Xu，最优插值网格](https://www.math.uci.edu/~chenlong/Papers/Chen.L%3BSun.P%3BXu.J2006.pdf) | 根据插值误差分配网格点及自适应节点已有研究。本轮用增添一条值域帽函数后辅助 Ritz 探针的能量投影误差下降量选择节点。 |

这里依据作者论文与出版商资料比对方法特征。对最近的扩散滤波工作可核查到摘要和章节节选。检索范围支持提出一个可复验的贡献假设，文献优先性和相对全部相关方法的优势仍待验证。

## 慢模态簇编码的直接先例

[Drmač 和 Gugercin 的 Q-DEIM](https://doi.org/10.1137/15M1019271) 用带列主元 QR 选择插值节点并分析基变换不变性。[Damle、Minden 和 Ying](https://doi.org/10.1093/imaiai/iay008) 已经用同类 QR 操作从特征嵌入中恢复簇归属。选点和基不变性均不能单独主张为贡献。这里的实验变量是指定锚点上的二维值，将一个低频特征值簇编码为两个标量场，再用值域帽函数生成稀疏的 Boomer 两层补偿。固定列预算的全簇展开和随机两坐标混合是直接实验对照。
