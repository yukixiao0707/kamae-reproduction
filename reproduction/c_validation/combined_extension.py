# Copyright [2024] Expedia, Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS.
"""训练隔离、未知类别与多算子组合的 CPU 扩展实验。"""

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds
import tensorflow as tf
from kamae.spark.estimators import StandardScaleEstimator
from kamae.spark.pipeline import KamaeSparkPipeline
from kamae.spark.transformers import BucketizeTransformer, MultiplyTransformer
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "results" / "combined_extension"
STRINGS = ["UserID", "MovieID", "Occupation", "Genres"]
INPUTS = STRINGS + ["EventDays32", "EventDays64"]
CATEGORICAL = [
    "UserID_indexed",
    "MovieID_indexed",
    "Occupation_indexed",
    "Genres_split",
    "Genres_indexed",
]
NUMERIC = [
    f"time_{stage}_{bits}" for bits in [32, 64] for stage in ["z", "scaled", "bucket"]
]
OUTPUTS = CATEGORICAL + NUMERIC
CONTINUOUS = [name for name in NUMERIC if "bucket" not in name]
META = ["row_id", "natural_unseen", "has_unseen", "injected"]
SPLITS = [-1.0, 0.0, 0.1, 1.0]
SENTINELS = {
    "MovieID": "C_UNSEEN_MOVIE",
    "Occupation": "C_UNSEEN_OCCUPATION",
    "Genres": "C_UNSEEN_GENRE",
}


def save_json(name: str, value: dict) -> None:
    """所有产物限制在本实验子目录。"""
    (OUT / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def sha(path: Path) -> str:
    """标记模型或原始输入的身份。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def params_snapshot(fitted: object) -> list:
    """序列化每个已拟合阶段的参数，便于核对参数冻结。"""
    return [
        {
            "class": type(stage).__name__,
            "uid": stage.uid,
            "params": {p.name: v for p, v in stage.extractParamMap().items()},
        }
        for stage in fitted.stages
    ]


def params_digest(params: list) -> str:
    """固定参数字典序后计算摘要。"""
    return hashlib.sha256(
        json.dumps(params, sort_keys=True, default=str).encode()
    ).hexdigest()


def make_inputs(model: object, data: dict) -> dict:
    """从模型输入声明构造张量，不猜测输入顺序。"""
    n = len(data["UserID"])
    return {
        t.name.split(":")[0]: tf.reshape(
            tf.constant(data[t.name.split(":")[0]], dtype=tf.as_dtype(t.dtype)),
            (n,) + tuple(int(d) if d is not None else 1 for d in t.shape[1:]),
        )
        for t in model.inputs
    }


def logical(values: np.ndarray, rows: int) -> np.ndarray:
    """保留样本轴，只去掉单元素包装维。"""
    if values.ndim == 0 or values.shape[0] != rows:
        raise ValueError(f"样本轴与行数不符：{values.shape}, {rows}")
    return values.reshape((rows,) + tuple(d for d in values.shape[1:] if d != 1))


def numeric_comparison(
    expected: np.ndarray, actual: np.ndarray, continuous: bool
) -> tuple:
    """区分连续容差、离散精确值和非有限值掩码，不相互豁免。"""
    ref, pred = expected.astype(np.float64), actual.astype(np.float64)
    nonfinite_bad = (
        (np.isnan(ref) != np.isnan(pred))
        | (np.isposinf(ref) != np.isposinf(pred))
        | (np.isneginf(ref) != np.isneginf(pred))
    )
    finite = np.isfinite(ref) & np.isfinite(pred)
    diff = np.zeros(ref.shape, dtype=np.float64)
    np.subtract(ref, pred, out=diff, where=finite)
    diff = np.abs(diff)
    exact = (expected == actual) | (np.isnan(ref) & np.isnan(pred))
    equal = (~nonfinite_bad) & (
        (~finite | (diff <= 1e-6 + 1e-6 * np.abs(ref))) if continuous else exact
    )
    return equal, exact, diff, nonfinite_bad


def prepare() -> None:
    """只在训练集拟合一次，冻结后为全部测试组生成参考输出。"""
    spec = importlib.util.spec_from_file_location(
        "a_movielens", REPO / "reproduction/scripts/run_movielens_pipeline.py"
    )
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    spark = original.create_spark_session()
    spark.sparkContext.setLogLevel("ERROR")
    started = time.perf_counter()
    try:
        frame = original.load_training_dataframe(spark)
        order = Window.orderBy(
            "Timestamp",
            F.col("UserID").cast("int"),
            F.col("MovieID").cast("int"),
            "Rating",
        )
        frame = (
            frame.withColumn("row_id", F.row_number().over(order))
            .withColumn(
                "EventDays64",
                (F.col("Timestamp").cast("double") - F.lit(946684800.0))
                / F.lit(86400.0),
            )
            .withColumn("EventDays32", F.col("EventDays64").cast("float"))
            .select("row_id", "Timestamp", *INPUTS)
            .cache()
        )
        count = frame.count()
        if count != 1000209:
            raise ValueError(f"MovieLens 数据不完整：{count}")
        train_n = int(count * 0.8)
        train = frame.filter(F.col("row_id") <= train_n).cache()
        test = frame.filter(F.col("row_id") > train_n).cache()
        train_count, test_count = train.count(), test.count()
        assert train_count == train_n and train_count + test_count == count
        assert train.join(test, "row_id").limit(1).count() == 0
        print(f"时间划分：训练 {train_count:,}，测试 {test_count:,}", flush=True)
        # 词表只从训练集取得，包括 A 的 Genres 词表构建步骤。
        pipeline = original.build_preprocessing_pipeline(train)
        stages = list(pipeline.getStages())
        for bits, alias in [(32, "float"), (64, "double")]:
            stages.extend(
                [
                    StandardScaleEstimator(
                        inputCol=f"EventDays{bits}",
                        outputCol=f"time_z_{bits}",
                        inputDtype=alias,
                        outputDtype=alias,
                        layerName=f"c_standard_{bits}",
                    ),
                    MultiplyTransformer(
                        inputCol=f"time_z_{bits}",
                        outputCol=f"time_scaled_{bits}",
                        inputDtype=alias,
                        outputDtype=alias,
                        mathFloatConstant=0.1,
                        layerName=f"c_multiply_{bits}",
                    ),
                    BucketizeTransformer(
                        inputCol=f"time_scaled_{bits}",
                        outputCol=f"time_bucket_{bits}",
                        inputDtype=alias,
                        outputDtype="bigint",
                        splits=SPLITS,
                        layerName=f"c_bucket_{bits}",
                    ),
                ]
            )
        print("仅用训练集拟合 11 阶段管道", flush=True)
        fitted = KamaeSparkPipeline(stages=stages).fit(train)
        frozen = params_snapshot(fitted)
        frozen_hash = params_digest(frozen)
        save_json(
            "fitted_parameters.json",
            {"train_rows": train_count, "stages": frozen, "sha256": frozen_hash},
        )
        schema = [{"name": name, "dtype": "string", "shape": (1,)} for name in STRINGS]
        schema += [
            {"name": "EventDays32", "dtype": "float32", "shape": (1,)},
            {"name": "EventDays64", "dtype": "float64", "shape": (1,)},
        ]
        model = fitted.build_keras_model(input_schema=schema, output_names=OUTPUTS)
        model_path = OUT / "combined_preprocessing.keras"
        model.save(model_path)
        loaded = tf.keras.models.load_model(model_path)
        probe = test.orderBy("row_id").select(*INPUTS).limit(128).collect()
        values = {name: [row[name] for row in probe] for name in INPUTS}
        tensors = make_inputs(model, values)
        before, after = model(tensors, training=False), loaded(tensors, training=False)
        reload_match = {
            name: bool(np.array_equal(before[name].numpy(), after[name].numpy()))
            for name in OUTPUTS
        }
        vocab = {
            name: [row[0] for row in train.select(name).distinct().collect()]
            for name in STRINGS[:3]
        }
        genres = fitted.stages[4].getLabelsArray()
        for name, token in SENTINELS.items():
            assert token not in (genres if name == "Genres" else vocab[name])
        is_unseen = {name: ~F.col(name).isin(vocab[name]) for name in STRINGS[:3]}
        is_unseen["Genres"] = F.exists(
            F.split(F.col("Genres"), r"\|"), lambda g: ~g.isin(genres)
        )
        unknown = (
            is_unseen["UserID"]
            | is_unseen["MovieID"]
            | is_unseen["Occupation"]
            | is_unseen["Genres"]
        )
        clean = (
            test.withColumn("natural_unseen", unknown)
            .withColumn("has_unseen", unknown)
            .withColumn("injected", F.lit(False))
            .cache()
        )
        clean.count()
        natural = {
            name: clean.filter(predicate).count()
            for name, predicate in is_unseen.items()
        }
        dates = {
            name: df.agg(
                F.min("Timestamp").alias("min"), F.max("Timestamp").alias("max")
            )
            .first()
            .asDict()
            for name, df in [("train", train), ("test", test)]
        }

        # 按测试序号确定注入，保持真实与人工未见类别可分别统计。
        def injected_frame(fields: list, rate: int, small: bool = False) -> DataFrame:
            current = (
                clean.filter(F.col("row_id") <= train_n + 4096) if small else clean
            )
            selected = F.pmod(F.col("row_id") - F.lit(train_n + 1), F.lit(100)) < rate
            current = current.withColumn("injected", selected).withColumn(
                "has_unseen", F.col("has_unseen") | selected
            )
            for name in fields:
                current = current.withColumn(
                    name,
                    F.when(selected, F.lit(SENTINELS[name])).otherwise(F.col(name)),
                )
            return current

        groups = [("real_test", clean, [])]
        groups += [
            (
                f"joint_oov_{rate:02d}",
                injected_frame(list(SENTINELS), rate),
                list(SENTINELS),
            )
            for rate in [1, 5, 10]
        ]
        groups += [
            (f"single_{field.lower()}_10", injected_frame([field], 10, True), [field])
            for field in SENTINELS
        ]
        records = []
        manifest = {
            "status": "preparing",
            "input_sha256": {
                p.name: sha(p) for p in (REPO / "reproduction/data/ml-1m").glob("*.dat")
            },
            "total_rows": count,
            "train_rows": train_count,
            "test_rows": test_count,
            "order": ["Timestamp", "int(UserID)", "int(MovieID)", "Rating"],
            "timestamps": dates,
            "natural_unseen_rows_by_field": natural,
            "vocabulary_sizes": {name: len(vocab[name]) for name in vocab}
            | {"Genres": len(genres)},
            "splits": SPLITS,
            "continuous_atol": 1e-6,
            "continuous_rtol": 1e-6,
            "model_sha256": sha(model_path),
            "reload_probe_rows": len(probe),
            "reload_match": reload_match,
            "frozen_parameters_sha256": frozen_hash,
            "fit_scope": "train_only",
            "groups": records,
        }
        save_json("manifest.json", manifest)
        for name, dataset, injected_fields in groups:
            counts = (
                dataset.agg(
                    F.count("*").alias("rows"),
                    F.sum(F.col("injected").cast("int")).alias("injected_rows"),
                    F.sum(F.col("natural_unseen").cast("int")).alias(
                        "natural_unseen_rows"
                    ),
                )
                .first()
                .asDict()
            )
            print(f"生成 {name}：{counts}", flush=True)
            path = OUT / f"spark_{name}.parquet"
            fitted.transform(dataset).select(*META, *INPUTS, *OUTPUTS).coalesce(
                1
            ).write.mode("overwrite").parquet(str(path))
            assert params_digest(params_snapshot(fitted)) == frozen_hash
            records.append(
                {
                    "name": name,
                    "parquet": str(path),
                    "injected_fields": injected_fields,
                    **counts,
                    "actual_injection_rate": counts["injected_rows"] / counts["rows"],
                }
            )
            save_json("manifest.json", manifest)
        manifest.update(
            status="prepared",
            parameters_unchanged=True,
            prepare_seconds=time.perf_counter() - started,
        )
        save_json("manifest.json", manifest)
        print("全部参考输出已生成；拟合参数保持冻结", flush=True)
    finally:
        spark.stop()


def compare() -> None:
    """从同一行的原始输入运行已保存模型，独立核对所有阶段。"""
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "prepared":
        raise RuntimeError("参考输出尚未完整生成")
    model_path = OUT / "combined_preprocessing.keras"
    assert sha(model_path) == manifest["model_sha256"]
    model = tf.keras.models.load_model(model_path)
    all_stats, examples, bucket_examples, group_rows = [], [], [], {}
    started = time.perf_counter()
    for group in manifest["groups"]:
        dataset = ds.dataset(
            group["parquet"], format="parquet", exclude_invalid_files=True
        )
        stats, seen_rows = {}, 0
        cohort_counts = {
            key: 0 for key in ["all", "seen_inputs", "natural_unseen", "injected"]
        }
        for batch in dataset.to_batches(
            columns=META + INPUTS + OUTPUTS, batch_size=4096
        ):
            data = batch.to_pydict()
            n = batch.num_rows
            outputs = model(make_inputs(model, data), training=False)
            injected = np.asarray(data["injected"], dtype=bool)
            natural = np.asarray(data["natural_unseen"], dtype=bool)
            unseen = np.asarray(data["has_unseen"], dtype=bool)
            masks = {
                "all": np.ones(n, dtype=bool),
                "seen_inputs": ~unseen,
                "natural_unseen": natural & ~injected,
                "injected": injected,
            }
            assert sum(int(m.sum()) for key, m in masks.items() if key != "all") == n
            for key, mask in masks.items():
                cohort_counts[key] += int(mask.sum())
            arrays = {}
            for name in OUTPUTS:
                expected_raw = np.asarray(data[name])
                actual_raw = outputs[name].numpy()
                expected, actual = logical(expected_raw, n), logical(actual_raw, n)
                if expected.shape != actual.shape:
                    raise ValueError(
                        f"{name} 形状不一致：{expected.shape}, {actual.shape}"
                    )
                if name == "Genres_split":
                    expected, actual = expected.astype(str), actual.astype(str)
                    equal = expected == actual
                    exact = equal
                    diff = np.zeros(expected.shape, dtype=np.float64)
                    nonfinite_bad = np.zeros(expected.shape, dtype=bool)
                else:
                    equal, exact, diff, nonfinite_bad = numeric_comparison(
                        expected, actual, name in CONTINUOUS
                    )
                arrays[name] = (expected, actual, equal, diff)
                for cohort, mask in masks.items():
                    if not np.any(mask):
                        continue
                    key = (cohort, name)
                    item = stats.setdefault(
                        key,
                        {
                            "group": group["name"],
                            "cohort": cohort,
                            "output": name,
                            "criterion": (
                                "atol=rtol=1e-6" if name in CONTINUOUS else "exact"
                            ),
                            "rows": 0,
                            "elements": 0,
                            "mismatches": 0,
                            "exact_mismatches": 0,
                            "nonfinite_mismatches": 0,
                            "max_abs_error": 0.0,
                            "abs_error_sum": 0.0,
                            "spark_dtype": str(dataset.schema.field(name).type),
                            "keras_dtype": str(actual_raw.dtype),
                            "spark_first_shape": list(expected_raw.shape),
                            "keras_first_shape": list(actual_raw.shape),
                        },
                    )
                    item["rows"] += int(mask.sum())
                    item["elements"] += int(expected[mask].size)
                    item["mismatches"] += int((~equal[mask]).sum())
                    item["exact_mismatches"] += int((~exact[mask]).sum())
                    item["nonfinite_mismatches"] += int(nonfinite_bad[mask].sum())
                    item["max_abs_error"] = max(
                        item["max_abs_error"], float(diff[mask].max())
                    )
                    item["abs_error_sum"] += float(diff[mask].sum())
                if (~equal).any() and len(
                    [
                        r
                        for r in examples
                        if r["group"] == group["name"] and r["output"] == name
                    ]
                ) < 5:
                    for index in np.argwhere(~equal)[:5]:
                        idx = tuple(index)
                        examples.append(
                            {
                                "group": group["name"],
                                "output": name,
                                "row_id": data["row_id"][idx[0]],
                                "spark": str(expected[idx]),
                                "keras": str(actual[idx]),
                                "abs_error": float(diff[idx]),
                                "injected": bool(injected[idx[0]]),
                                "raw_input": {k: data[k][idx[0]] for k in INPUTS},
                            }
                        )
            for bits in [32, 64]:
                exp, act, equal, _ = arrays[f"time_bucket_{bits}"]
                for idx in np.flatnonzero(~equal)[:5]:
                    if (
                        len(
                            [
                                r
                                for r in bucket_examples
                                if r["group"] == group["name"] and r["bits"] == bits
                            ]
                        )
                        >= 5
                    ):
                        break
                    numeric_stages = {}
                    for stage in ["z", "scaled"]:
                        a, b, valid, error = arrays[f"time_{stage}_{bits}"]
                        numeric_stages[stage] = {
                            "spark": float(a[idx]),
                            "keras": float(b[idx]),
                            "abs_error": float(error[idx]),
                            "within_tolerance": bool(valid[idx]),
                        }
                    bucket_examples.append(
                        {
                            "group": group["name"],
                            "bits": bits,
                            "row_id": data["row_id"][idx],
                            "input": data[f"EventDays{bits}"][idx],
                            "spark_bucket": int(exp[idx]),
                            "keras_bucket": int(act[idx]),
                            "stages": numeric_stages,
                        }
                    )
            seen_rows += n
        if seen_rows != group["rows"]:
            raise ValueError(f"{group['name']} 行数不完整：{seen_rows}")
        for item in stats.values():
            item["mean_abs_error"] = item.pop("abs_error_sum") / item["elements"]
            item["passed"] = item["mismatches"] == 0
        all_stats.extend(stats.values())
        group_rows[group["name"]] = {"rows": seen_rows, "cohorts": cohort_counts}
        print(f"已比较 {group['name']}：{seen_rows:,} 行", flush=True)
    primary = [r for r in all_stats if r["cohort"] == "all"]
    assert sha(model_path) == manifest["model_sha256"]
    summary = {
        "rows_by_group": group_rows,
        "batch_size": 4096,
        "seconds": time.perf_counter() - started,
        "model_unchanged": True,
        "all_outputs_passed": all(r["passed"] for r in primary),
        "categorical_outputs_passed": all(
            r["passed"] for r in primary if r["output"] in CATEGORICAL
        ),
        "failed_group_outputs": [r for r in primary if not r["passed"]],
        "examples": examples,
        "bucket_chain_examples": bucket_examples,
        "statistics": all_stats,
        "interpretation": "执行完成不代表所有数值通过；空分组不写通过统计，原始反例完整保留。",
    }
    save_json("comparison.json", summary)
    with (OUT / "stage_comparison.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_stats[0]))
        writer.writeheader()
        writer.writerows(all_stats)
    print(
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k
                not in [
                    "statistics",
                    "examples",
                    "bucket_chain_examples",
                    "failed_group_outputs",
                ]
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(
        f"超过验收阈值的组×阶段数：{len(summary['failed_group_outputs'])}", flush=True
    )


def main() -> None:
    """可分步准备参考结果、运行独立比较。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=["prepare", "compare", "all"],
        default=os.environ.get("KAMAE_COMBINED_PHASE", "all"),
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.phase in ["prepare", "all"]:
        prepare()
    if args.phase in ["compare", "all"]:
        compare()


if __name__ == "__main__":
    main()
