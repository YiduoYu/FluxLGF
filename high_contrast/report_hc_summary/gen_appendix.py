"""Build appendix_data.tex: every result table behind report_hc_summary.tex, read from the result files.
Run from anywhere with the hypre_env python. Inputs are read only.
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FD = HERE.parents[1]                 # FD_LGF/
FLUX = FD / "fluxmg"
SSR = FD / "gensolver" / "round3"
OUT = HERE / "appendix_data.tex"

CLS = {
    "constant": "常系数 Poisson",
    "aligned anisotropy, one strong axis": "对齐各向异性，一个强方向",
    "aligned anisotropy, one weak axis": "对齐各向异性，一个弱方向",
    "aligned anisotropy, stronger": "对齐各向异性，更强",
    "high contrast, log-normal": "高对比度，对数正态",
    "high contrast, conducting tubes": "高对比度，导电管道",
    "high contrast, insulating inclusions": "高对比度，绝缘夹杂",
    "high contrast, percolating": "高对比度，渗流两相",
    "3T, SC26_01-like": "3T，类 SC26\\_01",
    "3T, strong electron conduction": "3T，强电子传导",
    "3T, strong feedback": "3T，强反馈",
    "3T, oblate box": "3T，扁方块",
    "3T, graded transition": "3T，渐变过渡",
    "3T, 3 capsules": "3T，3 胶囊",
    "3T, 10 capsules": "3T，10 胶囊",
    "3T, 10 capsules, other seed": "3T，10 胶囊，另一种子",
    "3T, 10 graded boxes": "3T，10 渐变扁方块",
    "multigroup, AMG easy (like U13-2)": "多群，AMG 易（类 U13-2）",
    "multigroup, middle (like U92-1)": "多群，中（类 U92-1）",
    "multigroup, AMG hard (like U107-4)": "多群，AMG 难（类 U107-4）",
    "3T, 5 capsules, new seed": "3T，5 胶囊，新种子",
    "3T, 10 capsules, new seed": "3T，10 胶囊，新种子",
    "3T, prolate": "3T，长椭球",
    "3T, octahedra, 10 % field": "3T，八面体，10\\% 扰动",
    "3T, other coupling": "3T，新耦合参数",
    "multigroup, new parameters": "多群，新参数",
    "3T, 3 capsules, m = 128": "3T，3 胶囊，$m=128$",
    "3T, 10 capsules, m = 128": "3T，10 胶囊，$m=128$",
    "real SC26_01": "真实 SC26\\_01",
}


def tt(s):
    return "\\texttt{" + s.replace("_", "\\_") + "}"


def esc(s):
    rep = {"\\": "\\textbackslash{}", "_": "\\_", "%": "\\%", "&": "\\&", "#": "\\#", "$": "\\$",
           "^": "\\^{}", "~": "\\~{}", "<": "\\textless{}", ">": "\\textgreater{}", "{": "\\{", "}": "\\}"}
    return "".join(rep.get(ch, ch) for ch in s)


out = []
w = out.append


def longtable(spec, head, rows, caption, size="\\footnotesize", sep="4pt"):
    w("{" + size + "\\setlength{\\tabcolsep}{" + sep + "}")
    w("\\begin{longtable}{" + spec + "}")
    w("\\caption{" + caption + "}\\\\")
    w("\\toprule\n" + head + " \\\\\n\\midrule\n\\endfirsthead")
    w("\\toprule\n" + head + " \\\\\n\\midrule\n\\endhead")
    w("\\bottomrule\n\\endlastfoot")
    for r in rows:
        w(r if r.startswith(("\\midrule", "\\addlinespace")) else r + " \\\\")
    w("\\end{longtable}}")


def tables_from(path, keep):
    """Table environments of a tex file, by 0-based index."""
    t = re.findall(r"\\begin\{table\}.*?\\end\{table\}", Path(path).read_text(), re.S)
    return [t[i] for i in keep]


# ---------------------------------------------------------------- A, B: FD-LGF
RENAME = [("方法二", "FD-LGF"), ("M2 $w{=}", "FD-LGF $w{=}"), ("BAMG $\\theta", "BoomerAMG $\\theta")]


def renamed(src, dst):
    """Copy a table file from an older report with the current method names."""
    t = Path(src).read_text()
    for a, b in RENAME:
        t = t.replace(a, b)
    (HERE / dst).write_text(t)
    return dst


w("\\section{FD-LGF，教授的 $\\chi^x$ 系列（二维、三维）}")
w("数据来自 11 月 3 日的 $\\chi^x$ 实验（\\texttt{FD\\_LGF/report/}）。算例 A：$a=\\chi^x$；B：阶跃分层；C：$\\chi^x$ 加规则夹杂；D：$\\chi^x$ 加随机夹杂。"
  "$w$ 是局部精确解外环的宽度。A、B 没有特征区，只用背景逆，$w$ 不起作用：二维的结果填在 $w=1/16$ 一列，三维填在不带 $w$ 的“FD-LGF”一列。"
  "判据 PCG，$\\|r\\|/\\|f\\|\\le10^{-10}$，单线程，3 次取中位数。")
w("\\subsection{二维}\n\\input{" + renamed(FD / "report/tables_2d.tex", "tab_chix_2d.tex") + "}")
w("\\subsection{三维}\n\\input{" + renamed(FD / "report/tables_3d.tex", "tab_chix_3d.tex") + "}")
w("\\FloatBarrier")
w("\\section{FD-LGF，更一般的 10 类二维和 5 类三维系数}")
w("数据来自报告 01（\\texttt{FD\\_LGF/high\\_contrast/report\\_hc/}）。所有系数写成 $a=\\chi^g$，$\\chi\\in\\{10^2,10^4,10^6,10^8\\}$。"
  "二维 $n=1023$ 为确认实验，方案锁定 \\texttt{c2ff7672}；其余为探索性扫描。")
w("\\input{" + renamed(FD / "high_contrast/report_hc/tables_scan.tex", "tab_hc_scan.tex") + "}")
w("\\input{" + renamed(FD / "high_contrast/report_hc/tables_1023.tex", "tab_hc_1023.tex") + "}")
w("\\FloatBarrier")

# ---------------------------------------------------------------- C: bt-fd
w("\\section{bt-fd（三温）}")
w("判据：双双精度原残差 $\\|b-Ax\\|/\\|b\\|<10^{-8}$，时间含构建，单线程，独立进程。")
w("\\subsection{H57 相同精度多方法锁定实验（sha256 \\texttt{82dbdb84}）}")
for t in tables_from(FD / "regions/report/report_h57.tex", [0, 1]):
    w(t)
w("\\subsection{审计修复后的 SC26\\_01 配对复测和 H56 G8--G11 重跑（报告 24）}")
for t in tables_from(FD / "regions/report/report_audit_fix.tex", [1, 2, 4]):
    w(t)
w("\\FloatBarrier")

# ---------------------------------------------------------------- D: FluxMG
w("\\section{FluxMG}")
w("判据：双双精度原残差 $<10^{-8}$，时间含构建，单线程。多群算例是 Yue 等 2026 的类比，不是他们的矩阵。")


def parse_suite(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        m = re.match(r"^(\S+)\s+(.+?)\s{2,}(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)(?:\s+K=(\d+))?\s*$", line)
        if m:
            rows.append(m.groups())
    return rows


def suite_rows(rows, with_k=False):
    res = []
    for c, cls, fi, ai, fs, as_, r, *k in rows:
        cls = CLS.get(cls.strip(), esc(cls.strip()))
        rr = f"\\textbf{{{r}}}" if float(r) < 1 else r
        res.append(" & ".join([tt(c), cls, fi, ai, fs, as_, rr] + ([k[0] or "--"] if with_k else [])))
    return res


H7 = "算例 & 类别 & Flux 迭代 & AMG 迭代 & Flux (s) & AMG (s) & 比值"
w("\\subsection{第 1 轮最终版本，测试集（对 BoomerAMG $\\theta=0.25$，11 月 6 日）}")
longtable("llrrrrr", H7, suite_rows(parse_suite(FLUX / "final_suite.txt")),
          "测试集 19 类，SCORE 0.559，赢 17/19，零失败。AMG 取 3 遍最快，FluxMG 为单遍。来源 \\texttt{fluxmg/final\\_suite.txt}。")

txt = (FLUX / "final_holdout.txt").read_text()
fin, start = txt.split("--- final")[1].split("--- start-like")
w("\\subsection{第 1 轮留出集：最终版本与起点版本}")
KRE = r"^(\S+)\s+(.+?)\s{2,}(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+K=(\d+)"


def krows(text):
    return [m.groups() for m in (re.match(KRE, l) for l in text.splitlines()) if m]


longtable("llrrrrrr", H7 + " & 区域数 $K$", suite_rows(krows(fin), True),
          "最终版本（\\texttt{best/}），留出 SCORE 0.665，赢 10/10。来源 \\texttt{fluxmg/final\\_holdout.txt}。")
longtable("llrrrrrr", H7 + " & 区域数 $K$", suite_rows(krows(start), True),
          "起点版本（循环开始时的代码），留出 SCORE 0.556，赢 9/10。")

w("\\subsection{第 30 轮：多群分组修复}")
pr = []
for f in sorted((FLUX / "round2_grouping").glob("paired_*.txt"), key=lambda p: int(p.stem.split("_")[1])):
    t = f.read_text()
    sc = re.search(r"SCORE: ([\d.\-]+)", t).group(1)
    wi = re.search(r"WINS: (\S+)", t).group(1)
    fa = re.search(r"FAILS: (\S+)", t).group(1)
    pr.append(f"{f.stem.split('_')[1]} & {'修复后' if 'new' in f.stem else '修复前'} & {sc} & {wi} & {fa}")
longtable("rlrrr", "顺序 & 代码 & SCORE & 胜 & 失败", pr,
          "测试集交替配对运行（桌面负载约 3）。中位数：修复前 0.494，修复后 0.565。来源 \\texttt{fluxmg/round2\\_grouping/paired\\_*.txt}。")
gr = []
for line in (FLUX / "round2_grouping/grouping.txt").read_text().splitlines():
    m = re.match(r"^(\S+:\S+)\s+(\d+)\s+([\d.]+)\s+\|\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+\|\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+\|\s+(\S+)", line)
    if m:
        c, ai, as_, pi, ps, prr, ri, rs, rr, st = m.groups()
        gr.append(" & ".join([tt(c), ai, as_, pi, ps, prr, ri, rs, f"\\textbf{{{rr}}}" if float(rr) < 1 else rr, "\\texttt{" + st.replace("|", "\\textbar ") + "}"]))
longtable("lrrrrrrrrl", "算例 & AMG & (s) & 两两 & (s) & 比值 & 规则 & (s) & 比值 & 规则分组", gr,
          "分组留出集 14 个（规则实现前固定）。“两两”为原 AMFR 两两分组，“规则”为加探针后的分组；比值对 BoomerAMG $\\theta=0.25$。"
          "两两 SCORE 0.404（12/14），规则 SCORE 1.088（14/14）。分组一栏中 e 表示可由对角解消去的场，\\textbar{} 分隔各组。来源 \\texttt{round2\\_grouping/grouping.txt}。",
          size="\\scriptsize", sep="3pt")
longtable("llrrrrr", H7, suite_rows(parse_suite(FLUX / "round2_grouping/holdout.txt")),
          "修复后的第 1 轮留出集，SCORE 0.812，赢 10/10（同一负载下）。来源 \\texttt{round2\\_grouping/holdout.txt}。")
rr = []
for line in (FLUX / "round2_grouping/real.txt").read_text().splitlines():
    m = re.match(r"real run (\d+) (\S+) probe (\S+) its (\d+) \[(.*?)\] ok (\S+) rel (\S+) build (\S+) total (\S+)", line)
    if m:
        n, st, pb, its, each, ok, rel, b, tot = m.groups()
        rr.append(f"{n} & \\texttt{{{st.replace('|', chr(92) + 'textbar ')}}} & {pb} & {its} & {each} & {rel} & {b} & {tot}")
longtable("rlrrlrrr", "运行 & 分组 & 探针 & 总迭代 & 各次修正迭代 & 最终相对残差 & 构建 (s) & 总时间 (s)", rr,
          "修复后真实 SC26\\_01，3 次独立运行。来源 \\texttt{round2\\_grouping/real.txt}。")
w(tables_from(FD / "regions/report/report_fluxmg.tex", [3])[0])


def load(path, key):
    t, bad, its, joint = defaultdict(list), set(), {}, set()
    for line in Path(path).read_text().splitlines():
        q = json.loads(line)
        k = (q["case"], q[key])
        if not q["ok"]:
            bad.add(k)
            continue
        t[k].append(q["time"])
        its[k] = q["its"]
        if q.get("joint"):
            joint.add(k)
    med = {k: float(np.median(v)) for k, v in t.items() if k not in bad}
    return med, bad, its, joint


import sys
sys.path[:0] = [str(FLUX)]
from eval_fluxmg import CASES, HOLDOUT  # noqa: E402
from eval_grouping import CASES as GROUPING  # noqa: E402
CLASSOF = dict(CASES + HOLDOUT)
med, bad, its, joint = load(FLUX / "structured/results_20261109.jsonl", "method")
m2, b2, i2, _ = load(FLUX / "amg_configs.jsonl", "cfg")
CFGS = ["fast.25", "fast.5", "fast.7", "fast.9", "falgout.25", "falgout.5", "noagg.25", "noagg.5", "sys.u", "sys.n"]


def cell(k, med, bad, its, joint=()):
    if k in bad:
        return "失败"
    if k not in med:
        return "--"
    return f"{med[k]:.2f}/{its[k]}" + ("$^\\dagger$" if k in joint else "")


groups = (("测试集", [c for c, _ in CASES]), ("留出集", [c for c, _ in HOLDOUT]), ("分组留出集", list(GROUPING)), ("真实矩阵", ["real"]))
w("\\subsection{11 月 9 日预注册比较：FluxMG、hypre 结构多重网格和 BoomerAMG}")
w("每格为“时间 (s)/迭代”，2 次打乱顺序重复的中位数；任一次失败即记为失败。$^\\dagger$ 表示结构方法内部有一组多场用一次 BoomerAMG V-cycle（混合）。"
  "AMG 列为 $\\theta=0.25$；“AMG 最优”取 D.5 中每个算例最快的设置。比值 = FluxMG / 对照。来源 \\texttt{fluxmg/structured/results\\_20261109.jsonl}、\\texttt{fluxmg/amg\\_configs.jsonl}。")
for name, cs in groups:
    rows = []
    for c in cs:
        st = [med[(c, m)] for m in ("pfmg", "pfmg_rb", "smg") if (c, m) in med]
        fl = med.get((c, "flux"))
        best = min(((v, k[1]) for k, v in m2.items() if k[0] == c), default=None)
        rs = f"{fl / min(st):.2f}" if fl and st else ("结构均失败" if fl else "--")
        ra = f"{fl / best[0]:.2f}" if fl and best else "--"
        bcell = f"{best[0]:.2f}/{i2[(c, best[1])]} {best[1]}" if best else "--"
        rows.append(" & ".join([tt(c)] + [cell((c, m), med, bad, its, joint) for m in ("flux", "pfmg", "pfmg_rb", "smg", "amg")] + [bcell, rs, ra]))
    longtable("lrrrrrlrr", "算例 & FluxMG & PFMG & PFMG-rb & SMG & AMG .25 & AMG 最优（设置） & /结构 & /AMG",
              rows, name + "。", size="\\scriptsize", sep="3pt")
w("预注册预测 P5（FluxMG 在 $\\ge70\\%$ 的测试和留出算例上快于最好的结构方法）：59\\%，不成立。")

w("\\subsection{BoomerAMG 全部设置}")
w("每格为“时间 (s)/迭代”，同一 \\texttt{solve\\_ir} 路径；重复运行取中位数。粗体为该算例最快的设置。fast = HMIS 粗化 + 激进粗化，falgout = Falgout 粗化，noagg = 不用激进粗化，数字为强连接阈值 $\\theta$；sys.u/sys.n 为多场 systems 设置（未知量方式 / 节点方式），只对多场算例运行。来源 \\texttt{fluxmg/amg\\_configs.jsonl}。")
for name, cs in groups:
    rows = []
    for c in cs:
        vals = {cf: m2.get((c, cf)) for cf in CFGS}
        ok = [v for v in vals.values() if v is not None]
        b = min(ok) if ok else None
        cells = []
        for cf in CFGS:
            k = (c, cf)
            if k in b2:
                cells.append("失败")
            elif k in m2:
                s = f"{m2[k]:.2f}/{i2[k]}"
                cells.append(f"\\textbf{{{s}}}" if m2[k] == b else s)
            else:
                cells.append("--")
        rows.append(" & ".join([tt(c)] + cells))
    longtable("l" + "r" * len(CFGS), "算例 & " + " & ".join(CFGS), rows, name + "。", size="\\tiny", sep="2.5pt")

w("\\subsection{autoresearch 迭代记录}")
rows = []
for line in (FLUX / "autoresearch.jsonl").read_text().splitlines():
    q = json.loads(line)
    if q.get("type") != "result":
        continue
    sc = q.get("score")
    sc = "--" if sc is None else esc(str(sc))
    rows.append(f"{q['iteration']} & {sc} & {esc(str(q['status']))} & \\parbox[t]{{9.2cm}}{{\\raggedright {esc(q['description'])}}}")
longtable("rlll", "轮 & SCORE & 结果 & 改动", rows,
          "迭代记录原文（英文）。第 14--29 轮为 16 组同批参数扫描，合并为一行。第 31--33 轮在 11 月 9 日，未按 SCORE 判定。来源 \\texttt{fluxmg/autoresearch.jsonl}。",
          size="\\scriptsize", sep="3pt")
w("\\FloatBarrier")

# ---------------------------------------------------------------- E: SSR
w("\\section{SSR（锁定版本 G，11 月 8 日）}")
w("每行：SSR 迭代和时间；BoomerAMG \\texttt{fast.25}；每个算例最优的 BoomerAMG 设置（含 AIR 和 systems 设置）。比值 = SSR / 对照，失败按 8 倍计。"
  "判据与 FluxMG 相同（双双精度原残差 $<10^{-8}$），3 次打乱顺序重复。来源 \\texttt{gensolver/round3/final\\_*\\_G.log}。")
names = {"dev": "开发集（23）", "holdout": "留出集（14）", "weak": "BoomerAMG 弱项集 WEAK（11）",
         "weakhold": "弱项留出 WEAK\\_HOLD（6，曾用于找 bug）", "weakhold2": "弱项干净留出 WEAK\\_HOLD2（9）"}
for key, title in names.items():
    t = (SSR / f"final_{key}_G.log").read_text()
    rows = []
    for line in t.splitlines():
        m = re.match(r"^(\S+)\s+ssr\s+(\d+) its\s+(\S+) s \| (\S+)\s+(\d+)\s+(\S+)\s+(\S+) \| (\S+)\s+(\d+)\s+(\S+)\s+(\S+)", line)
        if m:
            c, si, ss, _, fi, fs, fr, bc, bi, bs, br = m.groups()
            ss = "失败" if ss == "None" else ss
            fs = "失败" if fs == "None" else fs
            bs = "失败" if bs == "None" else bs
            b = lambda r: f"\\textbf{{{r}}}" if float(r) < 1 else r  # noqa: E731
            rows.append(" & ".join([tt(c), si, ss, fi, fs, b(fr), tt(bc), bi, bs, b(br)]))
    summ = "；".join(esc(l) for l in t.splitlines() if l.startswith(("SCORE", "FAILS")))
    longtable("lrrrrrlrrr", "算例 & SSR & (s) & fast.25 & (s) & 比值 & 最优设置 & 迭代 & (s) & 比值", rows,
              title + "。" + summ.replace("SCORE vs fast.25", "对 fast.25 的 SCORE").replace("SCORE vs best per case", "对每例最优的 SCORE").replace("wins", "胜").replace("FAILS ssr", "SSR 失败"),
              size="\\scriptsize", sep="3pt")

OUT.write_text("\n".join(out) + "\n")
print("wrote", OUT, len(out))
