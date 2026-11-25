# Copyright [2024] Expedia, Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS.
"""C 的分桶扩展：独立导出模型，测试边界、精度和保存后重载。"""

import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    DoubleType,
    FloatType,
    IntegerType,
    StructField,
    StructType,
)

from kamae.spark.pipeline import KamaeSparkPipeline
from kamae.spark.transformers import BucketizeTransformer

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SPLITS = [0.0, 0.1, 1.0]


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    spark = (
        SparkSession.builder.master("local[1]")
        .appName("C-Bucketize-boundaries")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )
    rows = []
    try:
        for dtype, sql_type, alias in [
            ("float32", FloatType(), "float"),
            ("float64", DoubleType(), "double"),
        ]:
            # 相邻值在输入精度中产生，避免用相同的十进制文本掩盖差异。
            values = [(-2.0, "常规"), (0.05, "常规"), (0.5, "常规"), (2.0, "常规")]
            for boundary in SPLITS:
                value = np.asarray(boundary, dtype=dtype)
                values.extend(
                    [
                        (
                            float(
                                np.nextafter(value, np.asarray(-np.inf, dtype=dtype))
                            ),
                            "左邻",
                        ),
                        (float(value), "边界"),
                        (
                            float(np.nextafter(value, np.asarray(np.inf, dtype=dtype))),
                            "右邻",
                        ),
                    ]
                )
            values.extend(
                [
                    (float("-inf"), "无穷值"),
                    (float("inf"), "无穷值"),
                    (float("nan"), "NaN"),
                ]
            )
            dataframe = spark.createDataFrame(
                [(i, float(v)) for i, (v, _) in enumerate(values)],
                StructType(
                    [
                        StructField("id", IntegerType(), False),
                        StructField("x", sql_type, False),
                    ]
                ),
            )
            pipeline = KamaeSparkPipeline(
                stages=[
                    BucketizeTransformer(
                        inputCol="x",
                        outputCol="bucket",
                        inputDtype=alias,
                        outputDtype="bigint",
                        layerName=f"c_bucket_{dtype}",
                        splits=SPLITS,
                    )
                ]
            )
            fitted = pipeline.fit(dataframe)
            expected = fitted.transform(dataframe).orderBy("id").collect()
            model = fitted.build_keras_model(
                input_schema=[{"name": "x", "dtype": dtype, "shape": (1,)}]
            )
            shape = (len(values),) + tuple(
                1 if d is None else int(d) for d in model.inputs[0].shape[1:]
            )
            inputs = {
                "x": tf.reshape(
                    tf.constant([float(v) for v, _ in values], dtype=dtype), shape
                )
            }
            before = model(inputs, training=False)["bucket"].numpy().reshape(-1)
            model_path = RESULTS / f"c_bucketize_{dtype}.keras"
            model.save(model_path)
            loaded = tf.keras.models.load_model(model_path)
            actual = loaded(inputs, training=False)["bucket"].numpy().reshape(-1)
            for i, (value, kind) in enumerate(values):
                rows.append(
                    {
                        "dtype": dtype,
                        "case": kind,
                        "input_repr": repr(value),
                        "spark_bucket": int(expected[i]["bucket"]),
                        "keras_bucket": int(actual[i]),
                        "abs_error": abs(int(expected[i]["bucket"]) - int(actual[i])),
                        "parity": int(expected[i]["bucket"]) == int(actual[i]),
                        "reload_equal": int(before[i]) == int(actual[i]),
                    }
                )
    finally:
        spark.stop()
    with (RESULTS / "bucketize_cases.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "splits": SPLITS,
        "total_cases": len(rows),
        "parity_failures": sum(not r["parity"] for r in rows),
        "reload_failures": sum(not r["reload_equal"] for r in rows),
        "failures": [r for r in rows if not r["parity"]],
        "note": "保留所有反例；NaN/无穷值与有限边界分开解释，分桶整数差异不以浮点容差豁免。",
    }
    (RESULTS / "bucketize_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
