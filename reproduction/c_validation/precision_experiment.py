# Copyright [2024] Expedia, Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS.
"""精度感知导出的 Spark 正确性、真实管道回归与 CPU 开销实验。"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import kamae.spark.transformers as transformers
import keras
import numpy as np
import pyarrow.dataset as ds
import tensorflow as tf
from kamae.spark.pipeline import KamaeSparkPipeline, KamaeSparkPipelineModel
from kamae.spark.transformers import BucketizeTransformer
from precision_aware import PrecisionBucketizeLayer, analyse_bucketize, export_pipeline
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    DoubleType,
    FloatType,
    IntegerType,
    StructField,
    StructType,
)

HERE = Path(__file__).resolve().parent
OUT = HERE / "results/precision_aware"
MODES = ["original", "float64", "strict", "adaptive"]
PROFILES = {
    "legacy": [0.0, 0.1, 1.0],
    "safe_nonzero": [-2.0, -0.5, 0.5, 2.0],
    "decimal": [-0.7, -0.3, 0.1, 0.3, 0.7, 1.1],
    "tiny": sorted([-1e-310, -1e-45, 0.0, 1e-45, 1e-310]),
    "wide": [-1e100, -1e40, -1.0, 0.0, 1.0, 1e40, 1e100],
    "close_double": [
        0.1,
        float(np.nextafter(0.1, np.inf)),
        float(np.nextafter(np.nextafter(0.1, np.inf), np.inf)),
        0.5,
    ],
    "signed_zero_duplicate": [-0.0, 0.0, 1.0],
    "negative": [-1.1, -0.7, -0.3, 0.0],
}


def write_json(name: str, data: dict) -> None:
    (OUT / name).write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def corpus(splits: list, dtype: str) -> tuple[np.ndarray, list[str]]:
    """结构化值、随机数和随机位模式与边界邻域共同覆盖。"""
    rng = np.random.default_rng(20261124)
    values, kinds = [], []
    for array, label in [
        (np.linspace(-2, 2, 2048, dtype=dtype), "grid"),
        (rng.uniform(-4, 4, 2048).astype(dtype), "random_real"),
    ]:
        values.extend(array)
        kinds.extend([label] * len(array))
    integer = "uint32" if dtype == "float32" else "uint64"
    raw = rng.integers(0, np.iinfo(integer).max, size=1024, dtype=integer).view(dtype)
    values.extend(raw)
    kinds.extend(["random_ieee_bits"] * len(raw))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        for split in splits:
            value = np.asarray(split, dtype=dtype)
            values.extend(
                [
                    np.nextafter(value, np.asarray(-np.inf, dtype=dtype)),
                    value,
                    np.nextafter(value, np.asarray(np.inf, dtype=dtype)),
                ]
            )
            kinds.extend(["boundary_left", "boundary", "boundary_right"])
        smallest = np.nextafter(
            np.asarray(0.0, dtype=dtype), np.asarray(1.0, dtype=dtype)
        )
        values.extend(
            [
                -np.inf,
                -np.finfo(dtype).max,
                -smallest,
                -0.0,
                0.0,
                smallest,
                np.finfo(dtype).max,
                np.inf,
                np.nan,
                -np.nan,
            ]
        )
        kinds.extend(["special"] * 10)
    return np.asarray(values, dtype=dtype), kinds


def make_models(
    fitted: KamaeSparkPipelineModel, schema: list, names: list, prefix: str
) -> dict:
    """逐方法记录不能导出的情况，不把缺失结果当作正确。"""
    records = {}
    for mode in MODES:
        try:
            if mode == "original":
                model = fitted.build_keras_model(
                    input_schema=schema, output_names=names
                )
                plan = {"mode": mode, "strategy": "Kamae 原始导出"}
            else:
                model, plan = export_pipeline(
                    fitted, schema, output_names=names, mode=mode
                )
            path = OUT / f"{prefix}_{mode}.keras"
            model.save(path)
            records[mode] = {
                "status": "exported",
                "path": str(path),
                "sha256": sha(path),
                "plan": plan,
            }
        except (ValueError, TypeError, OverflowError, tf.errors.OpError) as error:
            records[mode] = {"status": "export_error", "error": str(error)}
        keras.backend.clear_session()
    return records


def export_saved_graphs(profiles: list) -> None:
    """保存独立 TensorFlow 图，英文工作路径兼容本机原生文件接口。"""
    exports = []
    for dtype in ["float32", "float64"]:
        profile = next(row for row in profiles if row["name"] == f"legacy_{dtype}")
        source = Path(profile["models"]["adaptive"]["path"])
        work = Path.home() / "Documents/Codex/work" / f"kc_precision_saved_{dtype}"
        model = keras.models.load_model(source)
        model.export(str(work))
        target = OUT / f"saved_adaptive_{dtype}"
        shutil.copytree(work, target, dirs_exist_ok=True)
        exports.append(
            {
                "dtype": dtype,
                "source_model_sha256": sha(source),
                "directory": str(target),
                "saved_model_sha256": sha(target / "saved_model.pb"),
            }
        )
        keras.backend.clear_session()
    write_json("savedmodel_exports.json", {"models": exports})


def prepare() -> None:
    """用真实 Spark 输出作为答案，测试族参数不按方法结果筛选。"""
    spark = (
        SparkSession.builder.master("local[1]")
        .appName("C-Precision-aware-reference")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    profiles = []
    try:
        for dtype, sql_type, alias in [
            ("float32", FloatType(), "float"),
            ("float64", DoubleType(), "double"),
        ]:
            for label, splits in PROFILES.items():
                name = f"{label}_{dtype}"
                values, kinds = corpus(splits, dtype)
                frame = spark.createDataFrame(
                    [(i, float(value)) for i, value in enumerate(values)],
                    StructType(
                        [
                            StructField("id", IntegerType(), False),
                            StructField("x", sql_type, False),
                        ]
                    ),
                )
                fitted = KamaeSparkPipeline(
                    stages=[
                        BucketizeTransformer(
                            inputCol="x",
                            outputCol="bucket",
                            inputDtype=alias,
                            outputDtype="bigint",
                            layerName="precision_bucket",
                            splits=splits,
                        )
                    ]
                ).fit(frame)
                expected = [
                    row["bucket"]
                    for row in fitted.transform(frame)
                    .select("id", "bucket")
                    .orderBy("id")
                    .collect()
                ]
                array_path = OUT / f"{name}.npz"
                np.savez(
                    array_path,
                    inputs=values,
                    expected=np.array(expected, dtype=np.int64),
                    kinds=np.array(kinds),
                )
                models = make_models(
                    fitted,
                    [{"name": "x", "shape": (1,), "dtype": dtype}],
                    ["bucket"],
                    name,
                )
                profiles.append(
                    {
                        "name": name,
                        "dtype": dtype,
                        "splits": splits,
                        "rows": len(values),
                        "array": str(array_path),
                        "models": models,
                        "plan": analyse_bucketize(splits, dtype),
                    }
                )
                write_json(
                    "corpus_manifest.json",
                    {"status": "preparing", "seed": 20261124, "profiles": profiles},
                )
                print(f"Spark 参考与导出完成：{name}，{len(values)} 行", flush=True)
    finally:
        spark.stop()
    # 从已经保存的训练参数重建管道，仅转换分桶层，不重新拟合训练数据。
    frozen_path = HERE / "results/combined_extension/fitted_parameters.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    stages = [
        getattr(transformers, stage["class"])(**stage["params"])
        for stage in frozen["stages"]
    ]
    fitted = KamaeSparkPipelineModel(stages)
    combined = json.loads(
        (HERE / "results/combined_extension/manifest.json").read_text(encoding="utf-8")
    )
    schema = [
        {"name": key, "shape": (1,), "dtype": "string"}
        for key in ["UserID", "MovieID", "Occupation", "Genres"]
    ]
    schema += [
        {"name": "EventDays32", "shape": (1,), "dtype": "float32"},
        {"name": "EventDays64", "shape": (1,), "dtype": "float64"},
    ]
    names = [
        "UserID_indexed",
        "MovieID_indexed",
        "Occupation_indexed",
        "Genres_split",
        "Genres_indexed",
    ]
    names += [
        f"time_{step}_{bits}" for bits in [32, 64] for step in ["z", "scaled", "bucket"]
    ]
    models = make_models(fitted, schema, names, "combined")
    write_json(
        "corpus_manifest.json",
        {
            "status": "prepared",
            "seed": 20261124,
            "profiles": profiles,
            "combined_models": models,
            "combined_groups": combined["groups"],
            "frozen_parameters_file_sha256": sha(frozen_path),
            "model_source": "原组合实验的训练参数；未重新拟合",
            "contract": "非空、非降序 double 阈值；float32/float64 密集输入；不表示 SQL null",
        },
    )
    print("参考集与四种方法的组合管道导出完成", flush=True)
    export_saved_graphs(profiles)


def compare_corpora(manifest: dict) -> dict:
    """逐案例核对整数答案，同时检查风险守卫是否覆盖原始差异。"""
    summaries, details = [], []
    old = list(
        csv.DictReader(
            (HERE / "results/bucketize_cases.csv").open(encoding="utf-8-sig")
        )
    )
    for profile in manifest["profiles"]:
        data = np.load(profile["array"])
        values, expected, kinds = data["inputs"], data["expected"], data["kinds"]
        predictions = {}
        for mode, record in profile["models"].items():
            if record["status"] != "exported":
                summaries.append(
                    {
                        "profile": profile["name"],
                        "method": mode,
                        "rows": len(values),
                        "status": "export_error",
                        "mismatches": None,
                        "max_bucket_error": None,
                        "fallback_rows": None,
                    }
                )
                continue
            model = keras.models.load_model(record["path"])
            x = tf.constant(values.reshape(-1, 1))
            actual = model({"x": x}, training=False)["bucket"].numpy().reshape(-1)
            predictions[mode] = actual
            layer = model.get_layer("precision_bucket")
            fallback = (
                int(layer.risk_mask(x).numpy().sum())
                if mode == "adaptive"
                else (len(values) if mode == "strict" else 0)
            )
            mismatch = actual != expected
            summaries.append(
                {
                    "profile": profile["name"],
                    "method": mode,
                    "rows": len(values),
                    "status": "compared",
                    "mismatches": int(mismatch.sum()),
                    "max_bucket_error": int(np.max(np.abs(actual - expected))),
                    "fallback_rows": fallback,
                }
            )
            # 原有 32 例作为单独保留的回归集，不仅依赖新生成的边界集合。
            if profile["name"].startswith("legacy_"):
                previous = [row for row in old if row["dtype"] == profile["dtype"]]
                tensor = tf.constant(
                    [[float(row["input_repr"])] for row in previous],
                    dtype=profile["dtype"],
                )
                legacy_out = (
                    model({"x": tensor}, training=False)["bucket"].numpy().reshape(-1)
                )
                summaries[-1]["previous_cases"] = len(previous)
                summaries[-1]["previous_mismatches"] = int(
                    np.sum(
                        legacy_out
                        != np.array([int(row["spark_bucket"]) for row in previous])
                    )
                )
            keras.backend.clear_session()
        for i, value in enumerate(values):
            row = {
                "profile": profile["name"],
                "dtype": profile["dtype"],
                "case": str(kinds[i]),
                "input_repr": repr(float(value)),
                "input_bits_hex": values[i : i + 1].tobytes().hex(),
                "spark_bucket": int(expected[i]),
            }
            row.update(
                {
                    mode: (
                        int(predictions[mode][i])
                        if mode in predictions
                        else "EXPORT_ERROR"
                    )
                    for mode in MODES
                }
            )
            details.append(row)
        print(f"正确性核对完成：{profile['name']}", flush=True)
    with (OUT / "correctness_cases.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    write_json(
        "correctness.json",
        {
            "profiles": summaries,
            "total_inputs": len(details),
            "reference": "原 Spark BucketizeTransformer 实际输出",
        },
    )
    return {"rows": len(details), "profiles": summaries}


def compare_real(manifest: dict) -> dict:
    """七组已冻结训练参数的真实参考输出，检查修改是否产生回归。"""
    continuous = [
        f"time_{step}_{bits}" for bits in [32, 64] for step in ["z", "scaled"]
    ]
    keys = ["UserID", "MovieID", "Occupation", "Genres", "EventDays32", "EventDays64"]
    records = []
    for mode, saved in manifest["combined_models"].items():
        if saved["status"] != "exported":
            raise RuntimeError(f"组合模型导出失败 {mode}: {saved}")
        model = keras.models.load_model(saved["path"])
        names = list(model.output)
        for group in manifest["combined_groups"]:
            dataset = ds.dataset(
                group["parquet"], format="parquet", exclude_invalid_files=True
            )
            stats = {
                name: {
                    "method": mode,
                    "group": group["name"],
                    "output": name,
                    "rows": 0,
                    "elements": 0,
                    "mismatches": 0,
                    "max_abs_error": 0.0,
                    "abs_error_sum": 0.0,
                }
                for name in names
            }
            for batch in dataset.to_batches(columns=keys + names, batch_size=4096):
                data, n = batch.to_pydict(), batch.num_rows
                tensors = {
                    t.name.split(":")[0]: tf.constant(
                        np.asarray(data[t.name.split(":")[0]]).reshape(n, 1),
                        dtype=tf.as_dtype(t.dtype),
                    )
                    for t in model.inputs
                }
                actual = model(tensors, training=False)
                for name in names:
                    ref = np.asarray(data[name])
                    got = actual[name].numpy()
                    ref = ref.reshape((n,) + tuple(d for d in ref.shape[1:] if d != 1))
                    got = got.reshape((n,) + tuple(d for d in got.shape[1:] if d != 1))
                    if ref.shape != got.shape:
                        raise ValueError(f"逻辑形状不符：{name}")
                    if name == "Genres_split":
                        equal = ref.astype(str) == got.astype(str)
                        diff = np.zeros(ref.shape)
                    else:
                        a, b = ref.astype(np.float64), got.astype(np.float64)
                        diff = np.abs(a - b)
                        equal = (
                            diff <= (1e-6 + 1e-6 * np.abs(a))
                            if name in continuous
                            else ref == got
                        )
                    item = stats[name]
                    item["rows"] += n
                    item["elements"] += ref.size
                    item["mismatches"] += int((~equal).sum())
                    item["max_abs_error"] = max(
                        item["max_abs_error"], float(diff.max())
                    )
                    item["abs_error_sum"] += float(diff.sum())
            for item in stats.values():
                assert item["rows"] == group["rows"]
                item["mean_abs_error"] = item.pop("abs_error_sum") / item["elements"]
                records.append(item)
            print(f"组合回归完成：{mode}/{group['name']}", flush=True)
        keras.backend.clear_session()
    with (OUT / "real_pipeline_comparison.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    write_json(
        "real_pipeline.json",
        {
            "rows_per_method": sum(g["rows"] for g in manifest["combined_groups"]),
            "all_within_criteria": all(r["mismatches"] == 0 for r in records),
            "statistics": records,
        },
    )
    return {"all_within_criteria": all(r["mismatches"] == 0 for r in records)}


def benchmark(manifest: dict) -> dict:
    """相同 CPU 线程和 tf.function，预热后轮换方法次序并同步取回输出。"""
    rows = []
    rng = np.random.default_rng(20261124)
    rounds, iterations, warmup = 7, 50, 10
    for dtype in ["float32", "float64"]:
        for profile_name in ["safe_nonzero", "legacy"]:
            profile = next(
                p
                for p in manifest["profiles"]
                if p["name"] == f"{profile_name}_{dtype}"
            )
            models, functions = {}, {}
            for mode in MODES:
                models[mode] = keras.models.load_model(profile["models"][mode]["path"])
                functions[mode] = tf.function(
                    lambda x, model=models[mode]: model({"x": x}, training=False)[
                        "bucket"
                    ],
                    input_signature=[
                        tf.TensorSpec([None, 1], dtype=tf.as_dtype(dtype))
                    ],
                )
            for n in [1, 128, 4096]:
                rates = [0.0] if profile_name == "safe_nonzero" else [0.0, 0.01, 0.1]
                for rate in rates:
                    inputs = rng.uniform(-4, 4, (n, 1)).astype(dtype)
                    injected = int(n * rate)
                    if injected:
                        inputs[:injected, 0] = -np.nextafter(
                            np.asarray(0.0, dtype=dtype), np.asarray(1.0, dtype=dtype)
                        )
                    x = tf.constant(inputs)
                    fallback = int(
                        models["adaptive"]
                        .get_layer("precision_bucket")
                        .risk_mask(x)
                        .numpy()
                        .sum()
                    )
                    timing = {mode: [] for mode in MODES}
                    for mode in MODES:
                        for _ in range(warmup):
                            functions[mode](x).numpy()
                    for repeat in range(rounds):
                        order = (
                            MODES[repeat % len(MODES) :] + MODES[: repeat % len(MODES)]
                        )
                        for mode in order:
                            start = time.perf_counter_ns()
                            for _ in range(iterations):
                                functions[mode](x).numpy()
                            timing[mode].append(
                                (time.perf_counter_ns() - start) / iterations / 1000.0
                            )
                    for mode in MODES:
                        samples = np.array(timing[mode])
                        rows.append(
                            {
                                "dtype": dtype,
                                "profile": profile_name,
                                "batch_size": n,
                                "requested_boundary_rate": rate,
                                "actual_boundary_rows": injected,
                                "fallback_rows": fallback,
                                "fallback_rate": fallback / n,
                                "method": mode,
                                "median_us": float(np.median(samples)),
                                "p25_us": float(np.quantile(samples, 0.25)),
                                "p75_us": float(np.quantile(samples, 0.75)),
                                "round_mean_us": samples.tolist(),
                                "rounds": rounds,
                                "iterations_per_round": iterations,
                                "warmup": warmup,
                            }
                        )
            keras.backend.clear_session()
        print(f"CPU 基准完成：{dtype}", flush=True)
    write_json(
        "benchmark.json",
        {
            "measure": "tf.function 预热后，包含调用与 .numpy() 同步取回；排除加载、追踪、输入生成和文件 IO",
            "scope": "单机 CPU 微基准，非集群或真实在线延迟",
            "order": "每轮旋转方法顺序",
            "statistics": rows,
        },
    )
    with (OUT / "benchmark.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {"workload_method_rows": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=["test", "prepare", "compare", "export", "all"],
        default=os.environ.get("KAMAE_PRECISION_PHASE", "all"),
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.phase == "test":
        import pytest

        raise SystemExit(
            pytest.main(
                [str(HERE / "test_precision_aware.py"), "-q", "-p", "no:cacheprovider"]
            )
        )
    if args.phase in ["prepare", "all"]:
        prepare()
    if args.phase == "export":
        manifest = json.loads(
            (OUT / "corpus_manifest.json").read_text(encoding="utf-8")
        )
        export_saved_graphs(manifest["profiles"])
    if args.phase in ["compare", "all"]:
        manifest = json.loads(
            (OUT / "corpus_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["status"] == "prepared"
        correctness = compare_corpora(manifest)
        real = compare_real(manifest)
        performance = benchmark(manifest)
        write_json(
            "run_summary.json",
            {"correctness": correctness, "real": real, "performance": performance},
        )


if __name__ == "__main__":
    main()
