# Copyright [2024] Expedia, Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS.
"""独立逐批比较全量 Spark Parquet 和重新加载的 Keras 模型。"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds
import tensorflow as tf

import kamae  # noqa: F401  # 导入项目以注册自定义层。

HERE = Path(__file__).resolve().parent
INPUTS = ["UserID", "MovieID", "Occupation", "Genres"]
OUTPUTS = ["UserID_indexed", "MovieID_indexed", "Occupation_indexed", "Genres_indexed"]


def logical_values(values: np.ndarray, rows: int) -> np.ndarray:
    """仅移除明确的单元素包装维度，不改变向量特征顺序。"""
    if values.shape[0] != rows:
        raise ValueError(f"样本维错误：{values.shape}，期望 {rows}")
    shape = tuple(d for d in values.shape[1:] if d != 1)
    return values.reshape((rows,) + shape)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parquet", type=Path, default=HERE.parent / "outputs" / "spark_output.parquet"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=HERE.parent / "outputs" / "preprocessing_model.keras",
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()
    result_dir = HERE / "results"
    result_dir.mkdir(exist_ok=True)
    model = tf.keras.models.load_model(args.model)
    dataset = ds.dataset(args.parquet, format="parquet", exclude_invalid_files=True)
    stats = {
        name: {
            "column": name,
            "rows": 0,
            "elements": 0,
            "mismatches": 0,
            "max_abs_error": 0.0,
            "abs_error_sum": 0.0,
            "spark_type": str(dataset.schema.field(name).type),
        }
        for name in OUTPUTS
    }
    examples = []
    total = 0
    started = time.perf_counter()
    for batch in dataset.to_batches(
        columns=INPUTS + OUTPUTS, batch_size=args.batch_size
    ):
        data = batch.to_pydict()
        count = batch.num_rows
        tensors = {}
        for tensor in model.inputs:
            name = tensor.name.split(":")[0]
            shape = (count,) + tuple(
                1 if d is None else int(d) for d in tensor.shape[1:]
            )
            tensors[name] = tf.reshape(tf.constant(data[name], dtype=tf.string), shape)
        outputs = model(tensors, training=False)
        if not isinstance(outputs, dict):
            raise TypeError("模型输出不是具名字典，拒绝猜测列顺序")
        for name in OUTPUTS:
            expected_raw = np.asarray(data[name])
            actual_raw = outputs[name].numpy()
            expected = logical_values(expected_raw, count)
            actual = logical_values(actual_raw, count)
            item = stats[name]
            if total == 0:
                item.update(
                    spark_shape=list(expected_raw.shape),
                    keras_shape=list(actual_raw.shape),
                    logical_shape=list(expected.shape[1:]),
                    keras_dtype=str(actual_raw.dtype),
                )
            if expected.shape != actual.shape:
                raise ValueError(
                    f"{name} 逻辑特征形状不一致：{expected.shape} != {actual.shape}"
                )
            # 本基线四个输出均为离散索引或 0/1 独热值，全部要求精确一致。
            equal = expected == actual
            difference = np.abs(expected.astype("float64") - actual.astype("float64"))
            item["rows"] += count
            item["elements"] += expected.size
            item["mismatches"] += int(np.count_nonzero(~equal))
            item["max_abs_error"] = max(
                item["max_abs_error"], float(np.max(difference))
            )
            item["abs_error_sum"] += float(np.sum(difference, dtype=np.float64))
            if not np.all(equal) and len(examples) < 20:
                for index in np.argwhere(~equal)[: 20 - len(examples)]:
                    idx = tuple(index)
                    examples.append(
                        {
                            "column": name,
                            "row_offset": total + int(index[0]),
                            "index": index.tolist(),
                            "spark": float(expected[idx]),
                            "keras": float(actual[idx]),
                            "input": {k: data[k][int(index[0])] for k in INPUTS},
                        }
                    )
        total += count
        if total % (args.batch_size * 20) == 0:
            print(f"已比较 {total:,} 行", flush=True)
    if total == 0:
        raise ValueError("输入为空，不允许空集通过")
    for item in stats.values():
        item["mean_abs_error"] = item.pop("abs_error_sum") / item["elements"]
        item["passed"] = item["mismatches"] == 0
    report = {
        "rows": total,
        "expected_movielens_rows": 1000209,
        "full_movielens_count_match": total == 1000209,
        "passed": all(x["passed"] for x in stats.values()) and total == 1000209,
        "seconds": time.perf_counter() - started,
        "batch_size": args.batch_size,
        "parquet": str(args.parquet),
        "model": str(args.model),
        "comparison": "精确比较四个离散输出；仅移除已记录的单元素包装维",
        "columns": list(stats.values()),
        "examples": examples,
    }
    (result_dir / "full_parity.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (result_dir / "full_parity.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(next(iter(stats.values())).keys())
        )
        writer.writeheader()
        writer.writerows(stats.values())
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
