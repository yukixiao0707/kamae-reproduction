# Copyright [2024] Expedia, Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS.
"""确认独立验证器不会静默吞掉样本维或特征维错误。"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "c_full_validation", Path(__file__).with_name("validate_full.py")
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
logical_values = _module.logical_values


def test_vector_order_is_preserved() -> None:
    """移除包装维时必须保留每行的六维类型特征。"""
    values = np.arange(18).reshape(3, 1, 6)
    actual = logical_values(values, 3)
    assert actual.shape == (3, 6)
    np.testing.assert_array_equal(actual, np.arange(18).reshape(3, 6))


def test_row_dimension_is_rejected() -> None:
    """元素总数相同也不能把两个样本硬改成三个样本。"""
    with pytest.raises(ValueError, match="样本维错误"):
        logical_values(np.arange(6).reshape(2, 3), 3)


def test_one_row_keeps_its_feature_axis() -> None:
    """最后一个分批只有一行时也不能丢失样本轴。"""
    assert logical_values(np.arange(6).reshape(1, 1, 6), 1).shape == (1, 6)
