# Copyright [2024] Expedia, Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS.
"""独立核对 IEEE 次序、风险守卫、原有 Spark 反例及保存重载。"""

import importlib.util
from pathlib import Path

import keras
import numpy as np
import pytest
import tensorflow as tf

_spec = importlib.util.spec_from_file_location(
    "precision_aware", Path(__file__).with_name("precision_aware.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("mode", ["strict", "adaptive"])
def test_existing_spark_cases(dtype: str, mode: str) -> None:
    """使用上轮 Spark 实测答案，不以本实现自造答案。"""
    import csv

    path = Path(__file__).parent / "results/bucketize_cases.csv"
    with path.open(encoding="utf-8-sig") as stream:
        rows = [r for r in csv.DictReader(stream) if r["dtype"] == dtype]
    x = tf.constant([float(r["input_repr"]) for r in rows], dtype=dtype)
    layer = _mod.PrecisionBucketizeLayer(
        splits=[0.0, 0.1, 1.0],
        precision_mode=mode,
        input_dtype=dtype,
        output_dtype="int64",
    )
    np.testing.assert_array_equal(
        layer(x).numpy(), [int(r["spark_bucket"]) for r in rows]
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_ieee_integer_order_and_zeros(dtype: str) -> None:
    """正负零相等，负数、无穷和负 NaN 次序正确。"""
    values = [-np.inf, -1.0, -0.0, 0.0, 1.0, np.inf, np.nan, -np.nan]
    keys = [_mod.ordered_key(v, dtype) for v in values]
    assert (
        keys[0] < keys[1] < keys[2] == keys[3] < keys[4] < keys[5] < keys[6] == keys[7]
    )


def test_float32_uses_upward_threshold() -> None:
    """阈值向下舍入时，Spark 比较的有效 float32 阈值必须向上取。"""
    p = _mod.analyse_bucketize([0.7], "float32")
    assert p["effective_thresholds"][0] > p["native_float32_thresholds"][0]


def test_plan_selects_safe_and_guarded_paths() -> None:
    assert (
        _mod.analyse_bucketize([-0.5, 0.5, 2.0], "float64")["strategy"] == "native_safe"
    )
    assert _mod.analyse_bucketize([0.0, 0.1], "float64")["strategy"] == "guarded"
    assert _mod.analyse_bucketize([1e-310], "float64")["strategy"] == "strict"


@pytest.mark.parametrize("bad", [[], [0.5, 0.1], [float("nan")]])
def test_invalid_thresholds_rejected(bad: list) -> None:
    with pytest.raises(ValueError):
        _mod.analyse_bucketize(bad, "float64")


def test_integer_dtype_not_silently_supported() -> None:
    with pytest.raises(ValueError):
        _mod.analyse_bucketize([0.0, 1.0], "int64")


def test_guarded_equals_strict_on_random_bits() -> None:
    """正负指数范围随机位输入，守卫路径必须与严格路径逐位输出相同。"""
    rng = np.random.default_rng(20261124)
    for dtype, integer in [("float32", "uint32"), ("float64", "uint64")]:
        raw = rng.integers(0, np.iinfo(integer).max, 4096, dtype=integer).view(dtype)
        x = tf.constant(raw)
        kwargs = dict(
            splits=[-0.7, -0.1, 0.0, 0.1, 0.7, 1.0],
            input_dtype=dtype,
            output_dtype="int64",
        )
        strict = _mod.PrecisionBucketizeLayer(precision_mode="strict", **kwargs)(
            x
        ).numpy()
        adaptive = _mod.PrecisionBucketizeLayer(precision_mode="adaptive", **kwargs)(
            x
        ).numpy()
        np.testing.assert_array_equal(adaptive, strict)


def test_save_reload_and_unseen_batch_shape() -> None:
    """保存后以不同批大小推理，验证自定义层配置完整。"""
    x = keras.Input(shape=(1,), dtype="float64", name="x")
    y = _mod.PrecisionBucketizeLayer(
        splits=[0.0, 0.1, 1.0],
        precision_mode="adaptive",
        input_dtype="float64",
        output_dtype="int64",
    )(x)
    model = keras.Model(x, y)
    directory = Path(__file__).parent / "results/precision_unit_models"
    directory.mkdir(exist_ok=True)
    path = directory / "precision.keras"
    model.save(path)
    loaded = keras.models.load_model(path)
    values = tf.constant([[-5e-324], [0.1], [0.10000000000000002], [2.0]], tf.float64)
    np.testing.assert_array_equal(loaded(values).numpy().reshape(-1), [1, 3, 3, 4])
