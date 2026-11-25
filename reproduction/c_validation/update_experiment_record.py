# Copyright [2024] Expedia, Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS.
"""从已保存的机器记录生成实验记录附录，不重新执行实验。"""

import csv
import json
import re
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
MARKER = (
    "<!-- 自动证据附录起点：由 update_experiment_record.py 从真实 JSON/CSV 更新 -->"
)


def read_json(path: Path) -> dict:
    """同时兼容 PowerShell 写入的 UTF-8 BOM。"""
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> None:
    """仅更新附录，保留人工撰写的实验过程。"""
    results = HERE / "results"
    full = read_json(results / "full_parity.json")
    assets = read_json(HERE / "assets_manifest.json")
    artifact = read_json(HERE / "handoff" / "产物清单.json")
    parts = [
        "## 附录 A. 实际运行时间",
        "以下记录均为 2026-11-23，北京时间（UTC+08:00）；耗时按已保存时间戳相减。",
        "| 步骤 | 开始时间 | 结束时间 | 过程耗时/秒 | 退出码 |",
        "|---|---|---|---:|---:|",
    ]
    for name in [
        "baseline",
        "original-check",
        "full-check",
        "extension",
        "probe",
        "plot",
        "diagnose",
    ]:
        record = read_json(results / f"{name}_run.json")
        # PowerShell 保存七位小数秒，Python 3.10 的解析器接受六位。
        start = datetime.fromisoformat(re.sub(r"(\.\d{6})\d+", r"\1", record["start"]))
        end = datetime.fromisoformat(re.sub(r"(\.\d{6})\d+", r"\1", record["end"]))
        parts.append(
            f'| {name} | {start:%H:%M:%S.%f} | {end:%H:%M:%S.%f} | {(end-start).total_seconds():.3f} | {record["exit_code"]} |'
        )
    parts.extend(
        [
            "## 附录 B. 全量逐字段统计",
            f'实际行数：{full["rows"]:,}；批大小：{full["batch_size"]}；完整数据量检查：{full["full_movielens_count_match"]}；总体判定：{full["passed"]}。',
            "| 字段 | 元素数 | 不一致数 | 最大绝对误差 | 平均绝对误差 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in full["columns"]:
        parts.append(
            f'| {row["column"]} | {row["elements"]:,} | {row["mismatches"]} | {row["max_abs_error"]:g} | {row["mean_abs_error"]:g} |'
        )
    parts.extend(
        [
            "下表形状来自第一批 4096 行；逻辑形状不包含样本轴，空元组表示每条记录一个标量。",
            "| 字段 | Spark/Arrow dtype | Keras dtype | Spark 原始形状 | Keras 原始形状 | 逻辑特征形状 |",
            "|---|---|---|---|---|---|",
        ]
    )
    for row in full["columns"]:
        parts.append(
            f'| {row["column"]} | `{row["spark_type"]}` | `{row["keras_dtype"]}` | `{tuple(row["spark_shape"])}` | `{tuple(row["keras_shape"])}` | `{tuple(row["logical_shape"])}` |'
        )
    parts.extend(
        [
            "## 附录 C. 全部 32 个扩展案例",
            "保留 CSV 顺序；一致/重载一致列分别对应跨框架比较和模型保存前后比较。",
            "| 编号 | 精度 | 案例类型 | 精确输入文本 | Spark | Keras | 桶索引绝对差 | 一致 | 重载一致 |",
            "|---:|---|---|---|---:|---:|---:|---|---|",
        ]
    )
    with (results / "bucketize_cases.csv").open(encoding="utf-8-sig") as stream:
        for index, row in enumerate(csv.DictReader(stream), start=1):
            parity = "是" if row["parity"] == "True" else "否"
            reload = "是" if row["reload_equal"] == "True" else "否"
            parts.append(
                f'| {index} | {row["dtype"]} | {row["case"]} | `{row["input_repr"]}` | {row["spark_bucket"]} | {row["keras_bucket"]} | {row["abs_error"]} | {parity} | {reload} |'
            )
    parts.extend(
        [
            "## 附录 D. 数据与产物身份",
            "### D.1 输入文件 SHA256",
            "| 文件 | SHA256 |",
            "|---|---|",
            f'| ml-1m.zip | `{assets["data"]["archive_sha256"]}` |',
        ]
    )
    for name, digest in assets["data"]["files"].items():
        parts.append(f"| {name} | `{digest}` |")
    parts.extend(
        [
            "### D.2 基线输出文件",
            "哈希标识本次具体产物；其他运行即使数值一致，也可能因封装元数据而有不同文件哈希。",
            "| 文件名 | 字节数 | SHA256 |",
            "|---|---:|---|",
        ]
    )
    for file in artifact["baseline_files"]:
        parts.append(
            f'| {Path(file["path"]).name} | {file["bytes"]:,} | `{file["sha256"]}` |'
        )
    parts.extend(
        [
            "## 附录 E. 实验结果文件",
            "以下文件对应正文各步骤的实际输出，可用于核对数据、过程和结论。",
            "| 证据 | 内容 |",
            "|---|---|",
        ]
    )
    evidence = {
        "baseline.log": "最终成功基线、数据数量、词表、schema 与样例",
        "original-check.log": "A 原始 128 行校验的实际输出",
        "full_parity.json": "全量数值、dtype、形状与通过标志",
        "full_parity.csv": "逐字段统计表",
        "extension.log": "扩展实验完整输出",
        "bucketize_cases.csv": "全部逐例原始结果",
        "bucketize_summary.json": "扩展汇总与四个反例",
        "bucketize_probe.json": "32 个底层算子复核结果和张量回读值",
        "environment.json": "最终环境、提交与原始数据哈希",
        "diagnose.log": "最终环境自检日志",
    }
    for name, meaning in evidence.items():
        if not (results / name).is_file():
            raise FileNotFoundError(results / name)
        parts.append(f"| [{name}](results/{name}) | {meaning} |")
    record_path = HERE / "实验记录.md"
    old = record_path.read_text(encoding="utf-8")
    prefix, found, _ = old.partition(MARKER)
    if not found:
        raise ValueError("未找到附录标记，拒绝覆盖实验正文")
    record_path.write_text(
        prefix
        + MARKER
        + "\n\n"
        + "\n\n".join(
            # Markdown 表格行之间不插入空行。
            "\n".join(block)
            for block in _paragraphs(parts)
        )
        + "\n",
        encoding="utf-8",
    )
    print("实验记录证据附录已更新")


def _paragraphs(lines: list[str]) -> list[list[str]]:
    """连续表格行放在同一段，其他段落分开。"""
    paragraphs = []
    for line in lines:
        if line.startswith("|") and paragraphs and paragraphs[-1][-1].startswith("|"):
            paragraphs[-1].append(line)
        else:
            paragraphs.append([line])
    return paragraphs


if __name__ == "__main__":
    main()
