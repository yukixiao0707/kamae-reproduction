# Copyright [2024] Expedia, Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS.
"""直接调用 TensorFlow 分桶算子，定位反例是否已经存在于底层算子。"""

import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

HERE = Path(__file__).resolve().parent


def main() -> None:
    """复查全部输入，不修改库、不替换实验结果。"""
    with (HERE / "results/bucketize_cases.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    results = []
    for dtype in ["float32", "float64"]:
        cases = [r for r in rows if r["dtype"] == dtype]
        tensor = tf.constant([float(r["input_repr"]) for r in cases], dtype=dtype)
        raw = tf.raw_ops.Bucketize(input=tensor, boundaries=[0.0, 0.1, 1.0]).numpy() + 1
        for r, bucket, observed in zip(cases, raw, tensor.numpy()):
            results.append(
                {
                    "dtype": dtype,
                    "input_repr": r["input_repr"],
                    "tensor_roundtrip": repr(float(observed)),
                    "raw_tf_bucket": int(bucket),
                    "keras_bucket": int(r["keras_bucket"]),
                    "same_as_keras": int(bucket) == int(r["keras_bucket"]),
                }
            )
    report = {
        "float32_boundary_0_1": float(np.float32(0.1)),
        "all_raw_tf_match_keras": all(r["same_as_keras"] for r in results),
        "cases": results,
    }
    (HERE / "results/bucketize_probe.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
