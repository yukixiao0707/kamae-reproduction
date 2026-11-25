# Copyright [2024] Expedia, Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS.
"""面向 Spark 浮点分桶语义的精度感知导出原型，仅支持 TensorFlow。"""

import math
from typing import Optional

import keras
import numpy as np
import tensorflow as tf
from kamae.keras.core.utils.input_utils import enforce_single_tensor_input
from kamae.keras.tensorflow.layers import BucketizeLayer
from kamae.spark.pipeline import KamaeSparkPipelineModel
from kamae.spark.transformers import BucketizeTransformer


def canonical_dtype(dtype: str) -> str:
    """将 Spark 别名转换为浮点名称；不静默支持其他输入类型。"""
    name = {"float": "float32", "double": "float64"}.get(dtype, dtype)
    if name not in ("float32", "float64"):
        raise ValueError(f"本原型仅处理显式 float32/float64 分桶，收到 {dtype!r}")
    return name


def ordered_key(value: float, dtype: str) -> int:
    """将 IEEE 浮点位映射为有序整数，统一正负零，并把 NaN 放到末尾。"""
    dtype = canonical_dtype(dtype)
    int_type = "int32" if dtype == "float32" else "int64"
    bits = int(np.asarray(value, dtype=dtype).view(int_type))
    maximum = np.iinfo(int_type).max
    magnitude = bits & int(maximum)
    inf_pattern = 0x7F800000 if dtype == "float32" else 0x7FF0000000000000
    if magnitude > inf_pattern:
        return int(maximum)
    if magnitude == 0:
        return 0
    return bits ^ int(maximum) if bits < 0 else bits


def effective_threshold(value: float, dtype: str) -> float:
    """float32 输入和 Spark double 字面量比较时，阈值取向上的可表示值。"""
    dtype = canonical_dtype(dtype)
    if dtype == "float64":
        return float(value)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        rounded = np.float32(value)
        if float(rounded) < value:
            rounded = np.nextafter(rounded, np.float32(np.inf))
    return float(rounded)


def analyse_bucketize(splits: list[float], dtype: str) -> dict:
    """分析阈值精度，选择可直接导出、输入守卫或全量兼容路径。"""
    dtype = canonical_dtype(dtype)
    values = [float(x) for x in splits]
    if not values or any(math.isnan(x) for x in values) or values != sorted(values):
        raise ValueError("阈值必须非空、非降序，且不能包含 NaN")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        native = [float(np.float32(x)) for x in values]
    effective = [effective_threshold(x, dtype) for x in values]
    risky_intervals = []
    for source, target in zip(native, effective):
        if source != target:
            keys = sorted([ordered_key(source, dtype), ordered_key(target, dtype)])
            risky_intervals.append(keys)
    # 原算子的 float 阈值溢出、重复或非正规阈值不交给快速路径。
    tiny = float(np.finfo(np.float32).tiny)
    special = any(not math.isfinite(x) or (0 < abs(x) < tiny) for x in values)
    special = (
        special
        or any(not math.isfinite(x) for x in native)
        or len(set(native)) != len(native)
    )
    has_zero = any(x == 0 for x in values)
    strategy = (
        "strict"
        if special
        else ("guarded" if risky_intervals or has_zero else "native_safe")
    )
    return {
        "dtype": dtype,
        "splits": values,
        "native_float32_thresholds": native,
        "effective_thresholds": effective,
        "threshold_keys": [ordered_key(x, dtype) for x in effective],
        "risk_intervals": risky_intervals,
        "guard_subnormal_at_zero": has_zero,
        "strategy": strategy,
        "special_threshold": special,
        "semantic_contract": "Spark 浮点输入与 double 阈值比较；NaN 高于所有数；正负零相等",
    }


@keras.saving.register_keras_serializable(package="CPrecision")
class PrecisionBucketizeLayer(BucketizeLayer):
    """整数位比较保持阈值语义，守卫路径只修正精度风险输入。"""

    jit_compatible = False

    def __init__(
        self: "PrecisionBucketizeLayer",
        splits: list[float],
        precision_mode: str = "adaptive",
        **kwargs: object,
    ) -> None:
        if precision_mode not in ("adaptive", "strict", "float64"):
            raise ValueError(precision_mode)
        super().__init__(splits=splits, **kwargs)
        self.precision_mode = precision_mode
        self.plan = analyse_bucketize(splits, self._input_dtype)

    @property
    def compatible_dtypes(self: "PrecisionBucketizeLayer") -> list[str]:
        return ["float32", "float64"]

    def keys(
        self: "PrecisionBucketizeLayer", values: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """位比较不会把非正规数交给会刷新为零的浮点比较指令。"""
        dtype = tf.int32 if self.plan["dtype"] == "float32" else tf.int64
        maximum = int(np.iinfo(dtype.as_numpy_dtype).max)
        bits = tf.bitcast(values, dtype)
        magnitude = tf.bitwise.bitwise_and(bits, tf.constant(maximum, dtype))
        inf_pattern = 0x7F800000 if dtype == tf.int32 else 0x7FF0000000000000
        key = tf.where(
            bits < 0, tf.bitwise.bitwise_xor(bits, tf.constant(maximum, dtype)), bits
        )
        key = tf.where(magnitude == 0, tf.zeros_like(key), key)
        key = tf.where(
            magnitude > tf.constant(inf_pattern, dtype),
            tf.fill(tf.shape(key), tf.constant(maximum, dtype)),
            key,
        )
        return key, magnitude

    def strict_values(self: "PrecisionBucketizeLayer", values: tf.Tensor) -> tf.Tensor:
        """每个阈值贡献一次整数比较，桶编号仍从 1 开始。"""
        keys, _ = self.keys(values)
        thresholds = tf.constant(self.plan["threshold_keys"], dtype=keys.dtype)
        count = tf.reduce_sum(tf.cast(keys[..., None] >= thresholds, tf.int64), axis=-1)
        return count + tf.constant(1, tf.int64)

    def risk_mask(self: "PrecisionBucketizeLayer", values: tf.Tensor) -> tf.Tensor:
        """报告输入路由；同时供混合执行和实验核对使用。"""
        if self.plan["strategy"] == "strict":
            return tf.ones(tf.shape(values), dtype=tf.bool)
        if self.plan["strategy"] == "native_safe":
            return tf.zeros(tf.shape(values), dtype=tf.bool)
        keys, magnitude = self.keys(values)
        mask = tf.zeros(tf.shape(keys), dtype=tf.bool)
        if self.plan["guard_subnormal_at_zero"]:
            normal_min_bits = (
                0x00800000 if keys.dtype == tf.int32 else 0x0010000000000000
            )
            mask = (magnitude > 0) & (
                magnitude < tf.constant(normal_min_bits, keys.dtype)
            )
        for low, high in self.plan["risk_intervals"]:
            mask = mask | (
                (keys >= tf.constant(low, keys.dtype))
                & (keys <= tf.constant(high, keys.dtype))
            )
        return mask

    @enforce_single_tensor_input
    def _call(
        self: "PrecisionBucketizeLayer", inputs: tf.Tensor, **kwargs: object
    ) -> tf.Tensor:
        if self.precision_mode == "float64":
            values = tf.cast(inputs, tf.float64)
            thresholds = tf.constant(self.splits, tf.float64)
            out = (
                tf.reduce_sum(
                    tf.cast(values[..., None] >= thresholds, tf.int64), axis=-1
                )
                + 1
            )
            # Spark 的 NaN 排在所有非 NaN 数值之后。
            _, magnitude = self.keys(inputs)
            inf_pattern = (
                0x7F800000 if self.plan["dtype"] == "float32" else 0x7FF0000000000000
            )
            out = tf.where(
                magnitude > tf.constant(inf_pattern, magnitude.dtype),
                tf.cast(len(self.splits) + 1, tf.int64),
                out,
            )
        elif self.precision_mode == "strict" or self.plan["strategy"] == "strict":
            out = self.strict_values(inputs)
        else:
            out = (
                tf.cast(
                    tf.raw_ops.Bucketize(input=inputs, boundaries=self.splits), tf.int64
                )
                + 1
            )
            if self.plan["strategy"] == "guarded":
                mask = tf.reshape(self.risk_mask(inputs), [-1])
                flat = tf.reshape(inputs, [-1])
                native = tf.reshape(out, [-1])

                def repair() -> tf.Tensor:
                    indices = tf.where(mask)
                    corrected = self.strict_values(tf.gather_nd(flat, indices))
                    return tf.tensor_scatter_nd_update(native, indices, corrected)

                out = tf.reshape(
                    tf.cond(tf.reduce_any(mask), repair, lambda: native),
                    tf.shape(inputs),
                )
        return self._cast(out, "int64")

    def get_config(self: "PrecisionBucketizeLayer") -> dict:
        config = super().get_config()
        config["precision_mode"] = self.precision_mode
        return config


class ExportBucketizeTransformer(BucketizeTransformer):
    """只替换 Keras 层工厂，Spark 参考转换继承原实现。"""

    def __init__(
        self: "ExportBucketizeTransformer",
        precision_mode: str = "adaptive",
        **kwargs: object,
    ) -> None:
        self.precision_mode = precision_mode
        super().__init__(**kwargs)

    def get_keras_layer(self: "ExportBucketizeTransformer") -> keras.layers.Layer:
        return PrecisionBucketizeLayer(
            name=self.getLayerName(),
            input_dtype=self.getInputKerasDtype(),
            output_dtype=self.getOutputKerasDtype(),
            splits=self.getSplits(),
            precision_mode=self.precision_mode,
        )


def export_pipeline(
    fitted: KamaeSparkPipelineModel,
    input_schema: list[dict],
    output_names: Optional[list[str]] = None,
    mode: str = "adaptive",
) -> tuple[keras.Model, dict]:
    """复制已拟合阶段，仅替换分桶导出，不重拟合或改变原 Spark 管道。"""
    stages, plans = [], []
    for stage in fitted.expand_pipeline_stages():
        if isinstance(stage, BucketizeTransformer):
            plan = analyse_bucketize(stage.getSplits(), stage.getInputKerasDtype())
            plan.update(
                layer=stage.getLayerName(),
                input_column=stage.getInputCol(),
                output_column=stage.getOutputCol(),
            )
            plans.append(plan)
            stage = ExportBucketizeTransformer(
                precision_mode=mode,
                inputCol=stage.getInputCol(),
                outputCol=stage.getOutputCol(),
                inputDtype=stage.getInputDtype(),
                outputDtype=stage.getOutputDtype(),
                layerName=stage.getLayerName(),
                splits=stage.getSplits(),
            )
        else:
            stage = stage.copy({})
        stages.append(stage)
    model = KamaeSparkPipelineModel(stages).build_keras_model(
        input_schema=input_schema, output_names=output_names
    )
    return model, {
        "mode": mode,
        "bucketize_layers": plans,
        "stage_count": len(stages),
        "scope": "显式 float32/float64 Bucketize 的导出；保留上游计算，不修正上游误差；SQL null 不在密集张量契约内",
    }
