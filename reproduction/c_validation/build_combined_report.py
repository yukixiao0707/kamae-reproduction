# Copyright [2024] Expedia, Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS.
"""从联合实验真实产物生成独立说明和表格，不修改 README。"""

import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds

HERE = Path(__file__).resolve().parent
OUT = HERE / "results/combined_extension"


def load(name: str) -> dict:
    """读取本实验机器结果。"""
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def table(headers: list, rows: list) -> str:
    """统一导出易核对的 Markdown 表格。"""
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
        + ["| " + " | ".join(str(v) for v in row) + " |" for row in rows]
    )


def update_main_record(text: str) -> None:
    """只更新统一记录中的组合实验章节，不再生成独立说明文档。"""
    record = HERE / "实验记录.md"
    content = record.read_text(encoding="utf-8")
    start = "<!-- 组合扩展正文开始 -->"
    end = "<!-- 组合扩展正文结束 -->"
    if start not in content or end not in content:
        raise ValueError("统一实验记录缺少组合章节标记，拒绝覆盖其他正文")
    lines = text.splitlines()[1:]
    converted = []
    for line in lines:
        if re.match(r"^### \d+\.\d+", line):
            line = re.sub(r"^### (\d+\.\d+)", r"#### 12.\1", line)
        elif re.match(r"^## \d+\.", line):
            line = re.sub(r"^## (\d+)\.", r"### 12.\1", line)
        converted.append(line)
    block = (
        start
        + "\n\n## 12. 组合扩展：训练集拟合与多算子新数据验证\n"
        + "\n".join(converted)
        + "\n\n"
        + end
    )
    left = content.index(start)
    right = content.index(end, left) + len(end)
    record.write_text(content[:left] + block + content[right:], encoding="utf-8")


def main() -> None:
    """保存每组结果，不把浮点容差通过写成零误差。"""
    m, s, fitted = (
        load("manifest.json"),
        load("comparison.json"),
        load("fitted_parameters.json"),
    )
    assert m["status"] == "prepared" and m["parameters_unchanged"]
    assert s["model_unchanged"]
    stats = [r for r in s["statistics"] if r["cohort"] == "all"]
    real = [r for r in stats if r["group"] == "real_test"]
    counts = [
        [
            g["name"],
            g["rows"],
            g["injected_rows"],
            f'{g["actual_injection_rate"]:.6%}',
            s["rows_by_group"][g["name"]]["cohorts"]["seen_inputs"],
            s["rows_by_group"][g["name"]]["cohorts"]["natural_unseen"],
        ]
        for g in m["groups"]
    ]
    group_results = []
    for group in m["groups"]:
        current = [r for r in stats if r["group"] == group["name"]]
        group_results.append(
            [
                group["name"],
                len(current),
                sum(r["passed"] for r in current),
                sum(r["mismatches"] for r in current),
                sum(
                    r["exact_mismatches"] for r in current if r["criterion"] != "exact"
                ),
            ]
        )
    # 单独核对未知类别在参考输出中的编码，Keras 同列已完成逐元素比较。
    semantics = []
    specs = [
        ("MovieID", "C_UNSEEN_MOVIE", "MovieID_indexed", "OOV=0"),
        ("Occupation", "C_UNSEEN_OCCUPATION", "Occupation_indexed", "全零独热向量"),
        ("Genres", "C_UNSEEN_GENRE", "Genres_indexed", "[1,0,0,0,0,0]"),
    ]
    for group in m["groups"]:
        dataset = ds.dataset(
            group["parquet"], format="parquet", exclude_invalid_files=True
        )
        for field, token, output, expected in specs:
            if field not in group["injected_fields"]:
                continue
            checked, failures = 0, 0
            for batch in dataset.to_batches(columns=[field, output], batch_size=8192):
                data = batch.to_pydict()
                mask = np.array(data[field]) == token
                if not mask.any():
                    continue
                values = np.array(data[output])[mask]
                valid = (
                    (values == 0)
                    if field == "MovieID"
                    else (
                        np.all(values == 0, axis=1)
                        if field == "Occupation"
                        else np.all(values == np.array([1, 0, 0, 0, 0, 0]), axis=1)
                    )
                )
                checked += int(mask.sum())
                failures += int((~valid).sum())
            assert checked == group["injected_rows"]
            semantics.append(
                {
                    "group": group["name"],
                    "field": field,
                    "rows": checked,
                    "expected_encoding": expected,
                    "violations": failures,
                }
            )
    (OUT / "unknown_semantics.json").write_text(
        json.dumps(
            {
                "checks": semantics,
                "passed": all(r["violations"] == 0 for r in semantics),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    params = [
        [r["params"]["inputCol"], r["params"]["mean"][0], r["params"]["stddev"][0]]
        for r in fitted["stages"]
        if r["class"] == "StandardScaleTransformer"
    ]
    text = f"""# C 组合扩展实验：训练集拟合、新数据验证与多算子组合

## 1. 本次增加了什么

本实验在 A 的五阶段类别管道基础上，同时完成两个扩展：训练集拟合后在独立测试数据上验证，以及在完整管道中增加连续的数值处理分支。目的在于检查冻结训练参数后，面对自然未见输入、人工未知类别与多算子组合，Spark 和 Keras 是否仍保持一致。

本实验模型与参考输出单独保存于 `results/combined_extension/`。全部运行使用 CPU、本机 Spark local[1]。以下依次说明实际的数据划分、管道构建、操作命令、比较方法和结果。

## 2. 数据准备和训练/测试划分

沿用 A 的 MovieLens 三表读取、关联代码，得到 {m["total_rows"]:,} 行。按 Timestamp、数值 UserID、数值 MovieID、Rating 升序建立稳定序号。前 floor(0.8N)={m["train_rows"]:,} 行作为训练集，其余 {m["test_rows"]:,} 行作为测试集。

代码实际检查了两部分行数之和等于全量，并通过 row_id 关联确认没有重叠。训练集时间戳范围为 {m["timestamps"]["train"]["min"]}—{m["timestamps"]["train"]["max"]}；测试集为 {m["timestamps"]["test"]["min"]}—{m["timestamps"]["test"]["max"]}。切分处同一秒的记录按后续排序键分开，因此两个范围端点可相同。

类别词表、均值和总体标准差全部只从训练集取得。测试数据只调用 transform 和导出的 Keras 模型，不重新拟合。固定换算 `(Timestamp−946684800)/86400` 将事件时间变为天数；换算常数不是从测试集估计。随后分别转换为 EventDays32（float32）和 EventDays64（float64）。评分不参与模型输入，也不进行下游预测训练。

## 3. 自然未见输入与人工注入

### 3.1 真实测试数据

训练集观察到的不同 UserID 为 {m["vocabulary_sizes"]["UserID"]}，MovieID 为 {m["vocabulary_sizes"]["MovieID"]}，Occupation 为 {m["vocabulary_sizes"]["Occupation"]}，Genres 为 {m["vocabulary_sizes"]["Genres"]}。其中 UserID 数量仅用于划分“训练中是否出现过”，原管道仍用哈希，不为用户建立可学习词表。

{table(["输入字段", "测试中未在训练集出现的记录数"], [[k, v] for k, v in m["natural_unseen_rows_by_field"].items()])}

真实测试集共有 {s["rows_by_group"]["real_test"]["cohorts"]["natural_unseen"]:,} 行至少包含一种自然未见输入。各字段数量可能重叠，不能相加作为唯一行数。多数是未见用户；这与查词表的 OOV 不同。真正涉及未见电影的真实记录有 {m["natural_unseen_rows_by_field"]["MovieID"]} 行，真实职业和类型在该划分中没有新增类别，因此另设计人工输入验证它们。

### 3.2 人工未知类别

设置三个训练中不存在的标记：MovieID=`C_UNSEEN_MOVIE`、Occupation=`C_UNSEEN_OCCUPATION`、Genres=`C_UNSEEN_GENRE`。代码在注入前检查标记不属于训练词表。

完整测试集分别按 1%、5%、10% 的设定比例联合替换这三个字段；选择规则为测试序号 mod 100 小于设定百分数。由于最后一组不足 100 行，实际比例略有偏差，下表记录实际数量。

另取按时间最早的 4096 条测试记录，分别只替换电影、职业或类型，做 10% 单字段控制。其余特征与参考记录保持对应。单字段控制用于核对各编码分支，不宣称覆盖全部测试行。

{table(["测试组", "行数", "人工注入行数", "实际比例", "全部输入已见行数", "自然未见且未注入行数"], counts)}

每组的后三类记录互斥：全部已见、自然未见且未注入、人工注入；行数之和等于该组行数。全体是另一个统计口径。自然未见且同时被注入的行归人工注入组，避免双计。

七组累计 {sum(g["rows"] for g in m["groups"]):,} 个测试行实例，来自重复使用同一测试集的不同干预；不是新增了这么多独立样本。

## 4. 完整的多算子管道

沿用 A 的五个阶段：UserID 哈希、MovieID 字符串索引、Occupation 独热编码、Genres 拆分、Genres 数组索引。A 的 build_preprocessing_pipeline 只接收训练子集，因此类型词表也只来自训练数据。

另增加两条三阶段分支，共 11 个处理阶段：

```text
原有四个类别输入 → A 的五阶段处理 → 四个最终类别特征及 Genres_split

EventDays32 → StandardScale(float32) → Multiply(0.1,float32) → Bucketize
EventDays64 → StandardScale(float64) → Multiply(0.1,float64) → Bucketize
```

标准化使用训练集均值与总体标准差；数值缩放常数为 0.1；分桶阈值固定为 [-1, 0, 0.1, 1]，桶编号从 1 开始，0 留作 padding。

实际学习到的标准化参数：

{table(["输入", "训练均值", "训练总体标准差"], params)}

导出模型时通过 output_names 同时保留原有类别分支的五个输出，以及两条数值分支各三个中间/最终输出。这样能观察标准化、乘法、分桶各阶段，而不是只检查最后的桶编号。

拟合后保存全部阶段参数，并记录摘要 `{m["frozen_parameters_sha256"]}`。生成各测试组参考输出后重新计算摘要，参数保持一致；比较时也检查模型文件 SHA256 未改变。模型保存前后对 128 行探针的全部 11 个输出精确一致，该探针用于保存重载核对，不替代完整测试集的验证。

## 5. 怎么运行

在已配置的本机环境中，从项目根目录执行：

```powershell
cd 'D:\\a-learn\\研究生\\课程\\大数据处理\\kamae-reproduction'
& .\\reproduction\\c_validation\\run_c.ps1 -Step combined-prepare
& .\\reproduction\\c_validation\\run_c.ps1 -Step combined-check
& .\\.venv-c310\\Scripts\\python.exe .\\reproduction\\c_validation\\build_combined_report.py
```

第一步读取原始数据、时间划分、仅训练集拟合、导出新模型，并生成七组 Spark 参考输出；第二步在不重新拟合的情况下，用已保存 Keras 模型分批比较全部组；第三步从实际结果更新本节和未知类别编码核对表。

代码为 `combined_extension.py`。输出路径均在 C 文件夹中，与原基线产物分开。重跑 prepare 会覆盖本组合扩展的输出，因此应保持该次导出模型与该次 Parquet 成对使用。跨电脑需按现有环境说明修改 run_c.ps1 中的运行组件路径。

## 6. 比较方法

每批 4096 行，同批读取原始输入、Spark 输出、row_id 和分组标记。按 Keras 输入名称、dtype 和 shape 构建输入，以同一行的参考输出作比较，不依赖不同文件的行顺序。

字符串拆分结果、类别索引、独热向量和桶编号采用精确比较。标准化与缩放采用 `abs(Spark−Keras) ≤ 1e-6+1e-6*abs(Spark)`；同时额外记录非精确元素数。两端非有限值掩码单独核对；实际测试输入未包含 null、NaN 或无穷，因此不据此声称已覆盖它们。

只去掉单元素包装维，保留样本轴和向量特征维。每组、每个输入分组、每个输出分别保存行数、元素数、dtype、形状、最大/平均误差、精确差异数与超过验收阈值数。空的输入分组不生成“通过”结果。

连续数值使用 float64 汇总差值，输入分支仍分别保留 float32 与 float64。若最终桶编号偏离，代码会保存对应行的标准化和缩放结果，以分析上游小误差是否越过分桶边界。

## 7. 实际结果

### 7.1 七组总体结果

{table(["测试组", "比较输出数", "按预定标准通过数", "超过验收标准元素数", "连续输出非精确元素数之和"], group_results)}

“通过”不等于“所有连续输出都零误差”。类别、字符串和桶索引在本次各组均精确一致；连续阶段按预设浮点容差判断。

### 7.2 真实测试集逐阶段结果

{table(["输出", "验收规则", "最大绝对误差", "平均绝对误差", "非精确元素数", "超出标准元素数"], [[r["output"], r["criterion"], f'{r["max_abs_error"]:.12g}', f'{r["mean_abs_error"]:.12g}', r["exact_mismatches"], r["mismatches"]] for r in real])}

完整分组与逐阶段数据见 `results/combined_extension/stage_comparison.csv`。数值分支没有出现桶编号偏差，因此本次真实数据及注入组中未观察到上游浮点误差导致的分桶结果变化。该结论限定在已测试数据上。

### 7.3 人工未知类别的实际编码

{table(["测试组", "字段", "核对行数", "预期编码", "不符合编码的行数"], [[r["group"], r["field"], r["rows"], r["expected_encoding"], r["violations"]] for r in semantics])}

本表核对 Spark 参考输出的实际未知类别编码；同列 Keras 输出已在前述全量比较中逐元素核对。MovieID 使用无 mask 的一个 OOV 索引 0；Occupation 的 dropUnseen=True 使未见职业输出全零；Genres 的 mask 为 0、OOV 为 1，单个未见类型加填充得到 [1,0,0,0,0,0]。这同时检查了“未知输入确实被送入并处理”，而不只观察程序是否完成。

## 8. 与原有结果如何衔接

原基线证明 A 的管道在全量 MovieLens 输入上的翻译一致。本次进一步使用仅训练集学习的参数，验证较晚发生的数据、自然新用户/新电影与人工未知类别，并检查完整数值链的中间阶段。

先前 32 个独立分桶边界案例中的 4 个反例仍然成立。本次真实时间数值没有重现那些特殊输入，组合实验通过不推翻已有反例，也不能推出所有分桶输入必然一致。

## 9. 可以写入报告的结论

在按时间划分的 MovieLens 实验中，仅使用 800,167 条训练记录拟合类别词表和标准化参数，然后在 200,042 条测试记录、三种人工未知类别比例与三个单字段控制组上，验证 11 阶段 Spark 管道及其 Keras 模型。类别、字符串和最终分桶输出精确一致，连续中间输出存在浮点差异，但均满足预先设定的容差。本次观察到的浮点差异没有改变最终桶编号。

这一结果扩展了原始复现的数据范围和算子组合，提供了训练参数冻结后处理新输入的验证证据；不涉及推荐准确率、真实线上业务分布或集群性能结论。

## 10. 文件对应关系

| 文件 | 内容 |
|---|---|
| 实验记录.md 第 12 节 | 组合实验的数据划分、操作方法、判定标准与结果 |
| combined_extension.py | 参考输出准备和独立全阶段比较代码 |
| results/combined_extension/manifest.json | 数据划分、自然未见数量、场景、注入比例、参数冻结检查 |
| results/combined_extension/fitted_parameters.json | 训练集学习到的全部阶段参数 |
| results/combined_extension/combined_preprocessing.keras | 本次独立导出模型 |
| results/combined_extension/spark_*.parquet/ | 七组参考输入与所有阶段 Spark 输出 |
| results/combined_extension/stage_comparison.csv | 全部分组、逐阶段误差与判定 |
| results/combined_extension/comparison.json | 比较结果和潜在反例 |
| results/combined_extension/unknown_semantics.json | 人工未知类别实际编码核对 |
| results/combined-prepare_run.json、combined-check_run.json | 两步执行时间和退出码 |
| results/combined_tests.log | 比较规则与形状处理的 6 项检查结果 |
"""
    update_main_record(text)
    with (OUT / "group_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "group",
                "outputs",
                "passed_outputs",
                "mismatches",
                "continuous_exact_mismatches",
            ]
        )
        writer.writerows(group_results)
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob("*.keras")
    }
    (OUT / "report_metadata.json").write_text(
        json.dumps(
            {
                "model_sha256": hashes,
                "comparison_passed": s["all_outputs_passed"],
                "unknown_semantics_passed": all(
                    r["violations"] == 0 for r in semantics
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("统一实验记录的组合章节与汇总表已更新；未修改 README")


if __name__ == "__main__":
    main()
