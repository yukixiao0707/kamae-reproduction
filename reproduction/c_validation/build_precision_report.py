# Copyright [2024] Expedia, Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS.
"""从精度感知实验结果更新三份现有文档与图表，不新建 Markdown。"""

import argparse
import json
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "results/precision_aware"
LABELS = {
    "original": "原始导出",
    "float64": "统一 float64",
    "strict": "严格兼容",
    "adaptive": "精度感知",
}
START = "<!-- 精度感知正文开始 -->"
END = "<!-- 精度感知正文结束 -->"


def read(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def table(headers: list, rows: list) -> str:
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
        + ["| " + " | ".join(str(v) for v in row) + " |" for row in rows]
    )


def figures() -> None:
    """单面板分别展示确定性错误数和 CPU 轮间分布，全部数据另存 CSV。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from audit_panel_alignment import require_matplotlib_panel_alignment

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"],
            "font.size": 8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    correctness = read("correctness.json")
    methods = list(LABELS)
    errors = [
        sum(
            row["mismatches"] or 0
            for row in correctness["profiles"]
            if row["method"] == mode
        )
        for mode in methods
    ]
    fig, ax = plt.subplots(figsize=(183 / 25.4, 80 / 25.4))
    ax.bar(
        range(4), errors, width=0.55, color=["#8b959e", "#d3924b", "#527ca3", "#398b83"]
    )
    for i, count in enumerate(errors):
        ax.annotate(
            str(count),
            (i, count),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    ax.set_xticks(range(4), list(LABELS.values()))
    ax.set_ylabel("与 Spark 不一致的输入实例数")
    ax.set_ylim(0, max(errors) * 1.15)
    ax.set_yticks([0, 200, 400, 600, 800, 1000])
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.17, top=0.94)
    require_matplotlib_panel_alignment(
        fig, json_out=str(OUT / "precision_correctness.alignment.json"), strict=True
    )
    fig.savefig(OUT / "precision_correctness.pdf")
    fig.savefig(OUT / "precision_correctness.svg")
    fig.savefig(OUT / "precision_correctness.png", dpi=600)
    plt.close(fig)
    data = read("benchmark.json")["statistics"]
    settings = [
        (dtype, profile, rate)
        for dtype in ["float32", "float64"]
        for profile, rate in [
            ("safe_nonzero", 0.0),
            ("legacy", 0.0),
            ("legacy", 0.01),
            ("legacy", 0.1),
        ]
    ]
    ticklabels = [
        "32安全",
        "32守卫0",
        "32守卫1",
        "32守卫10",
        "64安全",
        "64守卫0",
        "64守卫1",
        "64守卫10",
    ]
    fig, ax = plt.subplots(figsize=(183 / 25.4, 80 / 25.4))
    for j, mode in enumerate(methods):
        rows = [
            next(
                row
                for row in data
                if row["dtype"] == dtype
                and row["profile"] == profile
                and row["requested_boundary_rate"] == rate
                and row["batch_size"] == 4096
                and row["method"] == mode
            )
            for dtype, profile, rate in settings
        ]
        median = np.array([row["median_us"] for row in rows])
        lower = np.array([row["p25_us"] for row in rows])
        upper = np.array([row["p75_us"] for row in rows])
        ax.errorbar(
            np.arange(8) + (j - 1.5) * 0.15,
            median,
            yerr=[median - lower, upper - median],
            fmt=["o", "s", "^", "D"][j],
            linestyle="none",
            capsize=2,
            markersize=3.5,
            color=["#8b959e", "#d3924b", "#527ca3", "#398b83"][j],
            label=LABELS[mode],
        )
    ax.set_xticks(range(8), ticklabels)
    ax.set_ylabel("每次调用耗时（微秒）")
    ax.set_xlabel("精度与负载：守卫后数字为设定的边界输入百分比")
    ax.set_ylim(0, 700)
    ax.set_yticks([0, 100, 200, 300, 400, 500, 600, 700])
    fig.legend(loc="upper center", ncols=4, frameon=False, fontsize=8)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.22, top=0.8)
    require_matplotlib_panel_alignment(
        fig, json_out=str(OUT / "precision_cost.alignment.json"), strict=True
    )
    fig.savefig(OUT / "precision_cost.pdf")
    fig.savefig(OUT / "precision_cost.svg")
    fig.savefig(OUT / "precision_cost.png", dpi=600)
    plt.close(fig)


def section() -> str:
    """问题来源、实现、设置和结论均来自当前完成的实验。"""
    correctness, benchmark, real = (
        read("correctness.json"),
        read("benchmark.json"),
        read("real_pipeline.json"),
    )
    portable = read("portable_check.json")
    hardware = read("hardware.json")
    assert all(row["status"] == "compared" for row in correctness["profiles"])
    assert all(row["mismatches"] == 0 for row in portable["results"])
    example = {
        row["method"]: row
        for row in benchmark["statistics"]
        if row["dtype"] == "float64"
        and row["profile"] == "legacy"
        and row["requested_boundary_rate"] == 0.01
        and row["batch_size"] == 4096
    }
    results = []
    for mode in LABELS:
        rows = [row for row in correctness["profiles"] if row["method"] == mode]
        results.append(
            [
                LABELS[mode],
                len(rows),
                sum(row["rows"] for row in rows if row["status"] == "compared"),
                sum(row["mismatches"] or 0 for row in rows),
                sum(row.get("previous_mismatches", 0) for row in rows),
            ]
        )
    cost = []
    for dtype in ["float32", "float64"]:
        for profile, rate, label in [
            ("safe_nonzero", 0.0, "安全阈值"),
            ("legacy", 0.0, "守卫/普通输入"),
            ("legacy", 0.01, "守卫/约1%边界"),
            ("legacy", 0.1, "守卫/约10%边界"),
        ]:
            rows = {
                row["method"]: row
                for row in benchmark["statistics"]
                if row["dtype"] == dtype
                and row["profile"] == profile
                and row["requested_boundary_rate"] == rate
                and row["batch_size"] == 4096
            }
            cost.append(
                [dtype, label, rows["adaptive"]["fallback_rows"]]
                + [f'{rows[mode]["median_us"]:.2f}' for mode in LABELS]
            )
    return f"""# 精度感知导出方法与实验

## 1. 从整个复现与扩展实验得到的问题链

A 完成 MovieLens 五阶段管道、版本适配和 128 行校验后，C 在独立 CPU 环境重新运行该流程，并将比较扩大到全部 1,000,209 行、29,006,061 个输出元素，确认原基线四个输出完全一致。

C 继续围绕同一个跨框架一致性问题，完成训练/测试隔离、未知类别、多算子组合和数值边界等验证。整体实验表明，常规数据、冻结参数后的新数据和未知类别场景满足既定比较标准，但在特定浮点边界输入上出现了离散桶编号差异。四个已知反例的具体触发位置是数值分桶，原始基线和七组真实数据比较结果保持成立。

因此，本工作从“验证是否一致”推进到“对已发现的精度风险如何处理”：提出精度感知导出策略，以当前出现差异的浮点分桶算子为实现对象，增加阈值分析、运行时风险守卫和兼容计算路径，再检验正确性、原管道回归与 CPU 开销。

## 2. 问题机制与方法目标

Spark 参考实现按输入与 double 字面量阈值比较；原始 Keras 路径调用 TensorFlow Bucketize，其阈值采用 float 表示。阈值舍入、上下溢或相邻阈值合并可能改变离散决策。非正规数比较也可能与 Spark 观察到的结果不同，因此单独扩大输入 dtype 不足以解决全部反例。

目标是保留已拟合词表、均值和标准差，仅改变分桶算子的导出实现，使导出图在声明的输入契约内匹配 Spark 的比较语义，同时允许普通输入使用原生路径。本原型处理显式 float32/float64 密集输入、非空且非降序的阈值，拒绝 NaN 阈值；SQL null 不用 NaN 冒充。

## 3. 提出的方法：导出分析与风险输入路由

### 3.1 导出时分析阈值

转换模块读取已拟合阶段的输入精度、阈值与输出配置，不重新拟合。对 float64 输入保留原 double 阈值；对 float32 输入，将 Spark double 阈值 t 转为“不小于 t 的最小可表示 float32 值”，因为对于可表示的 float32 输入 x，两种比较 x≥t 与 x≥ceil32(t) 等价。

随后比较原生 float 阈值与上述有效阈值：无阈值变化且不涉及零附近风险时保留原生路径；阈值变化时生成风险区间；阈值上下溢、合并、重复或涉及特殊精度条件时采用全量兼容路径。

### 3.2 兼容路径采用 IEEE 有序整数比较

兼容层将浮点的 IEEE 位模式转换为有序整数键 K(x)：保留数值次序，统一正负零，并将 NaN 放在所有非 NaN 数之后，匹配本次 Spark 参考语义。每个阈值用相同方式转换，输出为：

`bucket(x) = 1 + Σ I[K(x) ≥ K(effective_threshold_j)]`

位模式与整数比较避免让极小数再次经过会刷新为零的浮点比较路径。该层由 TensorFlow 图操作组成，不依赖服务时的 Python 回调。

### 3.3 普通输入与风险输入分别执行

对需要守卫的层，运行时用有序键判断输入是否落入阈值变化区间；零阈值附近的非正规数也进入兼容路径。普通输入由原生算子处理，风险位置通过 gather、兼容计算与 scatter 修正。静态分析判断风险较特殊的层使用全量兼容路径。此路由并不保证每种负载都比全量兼容更快，实际开销另行测量。

### 3.4 转换模块与导出产物

实现位于 `precision_aware.py`：`analyse_bucketize` 生成分析结果，`PrecisionBucketizeLayer` 实现三种可选路径，`export_pipeline` 复制已拟合阶段，仅替换分桶层工厂。其他阶段保留原参数，原 Spark 参考实现和 A 的源码保持原样。

导出结果包括 Keras 模型与逐层分析清单。两个自适应分桶模型另导出为 TensorFlow SavedModel，在独立进程中仅加载 TensorFlow 运行原 32 个案例，结果全部一致，且未导入 Kamae 或本原型模块。该检查验证这两个独立分桶图的可执行性，不代表已经部署真实在线服务。

## 4. 实验设置

### 4.1 对比方法

| 方法 | 实现 | 作用 |
|---|---|---|
| 原始导出 | 原 Kamae BucketizeLayer | 原实现基线 |
| 统一 float64 | 分桶输入和阈值采用 float64 比较，其余阶段不改 | 检查“扩大精度”是否足够 |
| 严格兼容 | 所有输入走有序整数比较 | 检查兼容计算的正确性与代价 |
| 精度感知 | 静态分析加原生/风险/全量兼容路由 | 检查所提转换策略 |

### 4.2 正确性测试集

采用 8 组阈值：原反例阈值 [0,0.1,1]、可精确表示的非零阈值、常见小数、极小数、宽数值范围、相邻 double 阈值、正负零及重复阈值、负阈值。分别使用 float32 和 float64，共 16 个配置。

每个配置包含 2048 个结构化网格值、2048 个随机实数、1024 个随机 IEEE 位模式、每个阈值及其左右相邻可表示数、正负零、极值、无穷和 NaN。固定随机种子为 20261124，共 {correctness["total_inputs"]:,} 个配置内输入实例。答案来自实际运行的原 Spark Transformer，逐个比较桶编号；案例全部保存在 CSV，未按是否通过筛选。

原有 32 个案例作为单独回归集保留。由于其中包含已用于定位问题的反例，另增加的多阈值和随机输入用于检查方案是否只适配已知案例。

### 4.3 真实组合管道回归

读取前一轮只由 800,167 行训练数据学习的参数，重建并导出四种方法的 11 阶段管道，不重新拟合。使用原有七组 Spark 参考输出，每种方法检查 {real["rows_per_method"]:,} 个测试行实例。类别、字符串与桶编号精确比较；连续中间值沿用绝对、相对容差 1e-6。修改只作用于分桶导出，不补偿上游连续算子已经产生的误差。

### 4.4 CPU 开销

实验 CPU 为 {hardware["cpu_model"]}，系统可见 {hardware["logical_processors"]} 个逻辑处理器，Windows、Python 3.10.20、TensorFlow 2.21.0、Keras 3.12.4；OMP 与 TensorFlow 内部/算子间线程数均为 2，GPU 不参与。比较批大小 1、128、4096，精确非零阈值与需要守卫的 [0,0.1,1] 两种类型，后者分别加入设定 0%、1%、10% 的负非正规边界输入。

每个负载/方法预热 10 次，测量 7 轮，每轮 50 次；每轮旋转方法顺序，报告每轮平均耗时的中位数及四分位范围。计时使用 tf.function，包含调用和 .numpy() 同步取回，排除模型加载、图追踪、输入生成与文件 IO。共 96 条“负载×方法”记录；小批量实际注入数按 floor(N×比例) 计算，不能把设定比例当作实际比例。

## 5. 正确性与真实管道结果

{table(["方法", "配置数", "实际比较输入实例", "与 Spark 不同的桶编号数", "原32例的差异数"], results)}

原始导出在完整压力测试中出现 991 个差异；统一 float64 降为 53 个，但仍保留原 32 例中的 2 个反例；严格兼容与精度感知在全部 82,296 个输入实例上均未出现差异，原四个反例也全部通过。因此，所观测到的修正来自比较语义与阈值表示的处理，不能简单归结为统一使用更宽 dtype。

![四种方法正确性](precision_aware/precision_correctness.png)

图：全部 16 个配置汇总的离散错误计数，每种方法面对相同的 82,296 个输入实例。该集合包含主动构造的边界和特殊数值，计数不表示线上业务错误率；确定性计数不添加误差条。

四种方法在原七组组合场景中均满足相应比较标准（总体核对：{real["all_within_criteria"]}）。本次改进消除了额外压力测试中的离散决策差异，同时没有使现有真实管道发生验收回归。真实管道原本没有触发这些桶编号反例，因此不能将该回归通过解释为下游推荐效果提升。

## 6. CPU 开销结果与解释

下表列出批大小 4096 的全部 8 个工作负载，单位为微秒，数值为 7 轮均值的中位数。其余批大小、各轮原值和四分位范围保留在 benchmark.csv/JSON 中。

{table(["输入精度", "负载", "兼容路由行数", "原始", "统一64", "严格兼容", "精度感知"], cost)}

![四种方法CPU开销](precision_aware/precision_cost.png)

图：批大小 4096。点为 7 轮平均调用耗时的中位数，误差线为第 25—75 百分位，重复单位为本机计时轮次。横轴“守卫1/10”表示设定约 1%/10% 边界输入，实际路由行为 40/409、总行数 4096；同一方法与输入精度保持相同符号和颜色。全部 8 个该批大小负载都展示。

安全阈值下，自适应路径接近原始调用开销；需要风险检查或稀疏修正时，额外位操作、判断和 gather/scatter 带来代价。例如 float64、约 1% 边界负载中，自适应中位耗时为 {example["adaptive"]["median_us"]:.2f} 微秒，严格兼容为 {example["strict"]["median_us"]:.2f} 微秒；该负载没有体现自适应提速。实验支持正确性改进，并展示路径选择的开销差别，不支持“所有负载都更快”的结论。原始及统一64方法在相关边界上仍有错误，选择方案时需要同时看正确性与耗时，而不能只按延迟排序。

## 7. 实现覆盖与运行命令

当前原型将精度感知策略落实到 float32/float64 分桶导出，保留其他预处理阶段，不是所有 Spark 算子的通用等价证明。NaN 输入按本次 Spark 数值排序语义处理；NaN 阈值、空阈值和未声明的整数输入被拒绝，密集张量不表示 SQL null。新方法的贡献是从整个复现及扩展实验暴露的数值一致性问题出发，实现可执行转换与兼容路径，并给出正确性与开销证据。

在已配置的本机环境中，从项目根目录运行：

```powershell
& .\\reproduction\\c_validation\\run_c.ps1 -Step precision-test
& .\\reproduction\\c_validation\\run_c.ps1 -Step precision-prepare
& .\\reproduction\\c_validation\\run_c.ps1 -Step precision-check
& .\\reproduction\\c_validation\\run_c.ps1 -Step precision-export
& .\\reproduction\\c_validation\\run_c.ps1 -Step precision-portable
```

precision-prepare 用 Spark 生成参考答案并导出四种方法；precision-check 运行完整正确性、真实回归与 CPU 微基准；precision-test 检查有序比较、风险路径、原反例及序列化。precision-export 可以独立重建两个 SavedModel，precision-portable 在独立进程复核它们。方法和验证代码均在 C 文件夹，结果位于 `results/precision_aware/`。
"""


def replace_block(content: str, block: str, before: str) -> str:
    if START in content and END in content:
        left, right = content.index(START), content.index(END) + len(END)
        return content[:left] + block + content[right:]
    if before not in content:
        raise ValueError(f"未找到章节位置：{before}")
    return content.replace(before, block + "\n\n" + before, 1)


def update_draft_only() -> None:
    """供 finalize 调用，防止以后生成初稿时丢掉新方法章节。"""
    handoff = HERE / "handoff"
    target = handoff / "precision_aware"
    target.mkdir(exist_ok=True)
    for pattern in ("*.json", "*.csv", "*.png", "*.pdf", "*.svg"):
        for path in OUT.glob(pattern):
            shutil.copy2(path, target / path.name)
    for name in (
        "combined_adaptive.keras",
        "legacy_float32_adaptive.keras",
        "legacy_float64_adaptive.keras",
    ):
        shutil.copy2(OUT / name, target / name)
    record_figures = handoff / "results/precision_aware"
    record_figures.mkdir(parents=True, exist_ok=True)
    for name in ("precision_correctness.png", "precision_cost.png"):
        shutil.copy2(OUT / name, record_figures / name)
    p = handoff / "实验章节初稿.md"
    body = section().split("\n", 1)[1]
    body = re.sub(r"^### (\d+\.\d+)", r"#### \1", body, flags=re.M)
    body = re.sub(r"^## (\d+)\.", r"### \1", body, flags=re.M)
    block = START + "\n\n## 从复现与扩展到精度感知导出方法\n" + body + "\n\n" + END
    content = replace_block(p.read_text(encoding="utf-8"), block, "## 实验结论")
    content = content.replace(
        "# C 独立验证与扩展实验", "# Kamae 复现、扩展验证与精度感知导出实验", 1
    )
    summary = "<!-- 精度方法结论开始 -->\n在整个复现与扩展实验暴露的数值一致性问题基础上，实现精度感知导出模块。16 个配置、82,296 个输入实例中，原始导出有 991 个离散差异，统一 float64 有 53 个，严格兼容与精度感知均为 0；原七组组合管道未发生验收回归。CPU 结果体现了正确性与开销的权衡，自适应路由未在所有负载下提速。\n<!-- 精度方法结论结束 -->"
    if "<!-- 精度方法结论开始 -->" in content:
        content = re.sub(
            r"<!-- 精度方法结论开始 -->.*?<!-- 精度方法结论结束 -->",
            lambda _: summary,
            content,
            flags=re.S,
        )
    else:
        content = content.replace("## 实验结论", "## 实验结论\n\n" + summary, 1)
    p.write_text(content, encoding="utf-8")
    state_path = handoff / "交接状态.json"
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.exists()
        else {}
    )
    state.update(
        precision_method_added=True,
        chapter_draft="已包含完整问题链、实验设置、方法实现、正确性与 CPU 开销",
    )
    state.setdefault("b_draft_received", False)
    state.setdefault("review_done", False)
    state_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def update_documents() -> None:
    raw = section().split("\n", 1)[1]
    body = re.sub(r"^### (\d+\.\d+)", r"#### 13.\1", raw, flags=re.M)
    body = re.sub(r"^## (\d+)\.", r"### 13.\1", body, flags=re.M)
    body = body.replace("(precision_aware/", "(results/precision_aware/")
    block = (
        START
        + "\n\n## 13. 精度感知导出：从问题发现到方法实现与验证\n"
        + body
        + "\n\n"
        + END
    )
    path = HERE / "实验记录.md"
    content = path.read_text(encoding="utf-8")
    if START not in content:
        content = content.replace(
            "## 13. 产物与使用方式", "## 14. 产物与使用方式"
        ).replace("## 14. 实验结论", "## 15. 实验结论")
    content = replace_block(content, block, "## 14. 产物与使用方式")
    content = content.replace(
        "实验日期：2026 年 11 月 23 日；运行时间采用北京时间。",
        "原复现与组合扩展：2026 年 11 月 23 日；精度感知方法实验与文档更新：2026 年 11 月 24 日；运行时间采用北京时间。",
    )
    if "| 方法改进 |" not in content:
        content = content.replace(
            "| 数据和文字交接 |",
            "| 方法改进 | 实现阈值分析、风险守卫和兼容比较的导出模块 | 四方法正确性对照、真实管道回归和 CPU 开销 |\n| 数据和文字交接 |",
            1,
        )
    summary = "<!-- 精度方法总结开始 -->\n在整个复现与扩展实验发现的问题基础上，新增精度感知导出原型，以浮点分桶实现兼容路径。82,296 个压力测试输入实例上的离散差异由原始导出的 991 个降至 0，统一 float64 对照仍有 53 个；原有组合管道未出现验收回归。兼容路径存在 CPU 代价，测试未支持所有负载都提速的结论。具体实现与开销结果见第 13 节。\n<!-- 精度方法总结结束 -->\n\n"
    if "<!-- 精度方法总结开始 -->" in content:
        content = re.sub(
            r"<!-- 精度方法总结开始 -->.*?<!-- 精度方法总结结束 -->\n\n",
            lambda _: summary,
            content,
            flags=re.S,
        )
    else:
        content = content.replace(
            "<!-- 自动证据附录起点：", summary + "<!-- 自动证据附录起点：", 1
        )
    path.write_text(content, encoding="utf-8")
    # 使用用户已改名的文件，不重新生成旧文件名。
    path = HERE / "新增的工作说明.md"
    content = path.read_text(encoding="utf-8-sig")
    new = """<!-- 新增方法工作开始 -->
## 6. 提出并实现精度感知导出方法

在整个复现与扩展实验过程中，发现跨框架数值一致性在特定条件下会偏离。基于已观察到的离散决策差异，新增阈值精度分析、风险输入守卫和有序整数兼容比较，形成可导出的处理模块，以浮点分桶作为实现对象。

与原始导出、统一 float64、全量严格兼容进行对照：16 个配置、82,296 个测试输入实例中，原始导出有 991 个桶编号差异，统一 float64 有 53 个，严格兼容和精度感知均为 0。原四个反例全部通过；原七组组合管道未出现验收回归。

此外测量了 96 条负载与方法组合的 CPU 开销。兼容处理存在额外代价，自适应路径未在所有负载下胜出，因此结果表述为正确性改进与开销权衡。两个 SavedModel 在新进程中无需导入 Kamae 或本模块即可通过原 32 例验证。

这项工作新增了实际的转换策略与执行实现；完整问题链、实验设置、方法和结果已写入实验记录及章节初稿。
<!-- 新增方法工作结束 -->
"""
    if "<!-- 新增方法工作开始 -->" in content:
        content = re.sub(
            r"<!-- 新增方法工作开始 -->.*?<!-- 新增方法工作结束 -->",
            lambda _: new.strip(),
            content,
            flags=re.S,
        )
    else:
        content = content.replace("## 6. 实验结果整理", "## 7. 实验结果整理")
        content = content.replace("## 7. 实验结果整理", new + "\n## 7. 实验结果整理")
    path.write_text(content, encoding="utf-8")
    update_draft_only()
    shutil.copy2(HERE / "实验记录.md", HERE / "handoff/实验记录.md")
    print("已更新初稿、实验记录及用户改名后的新增工作说明")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args()
    if args.figures:
        figures()
    update_documents()


if __name__ == "__main__":
    main()
