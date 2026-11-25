# Copyright [2024] Expedia, Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS.
"""检查联合实验的验收标准不会误放行索引差异或非有限值差异。"""

import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "c_combined", Path(__file__).with_name("combined_extension.py")
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def test_discrete_index_does_not_use_relative_tolerance() -> None:
    """大索引相差 1 也不能按浮点相对容差放行。"""
    equal, _, _, _ = _module.numeric_comparison(
        np.array([1000000000]), np.array([1000000001]), False
    )
    assert not equal[0]


def test_continuous_and_exact_results_are_separate() -> None:
    """小误差可通过连续阈值，但仍需记录非精确元素。"""
    equal, exact, _, _ = _module.numeric_comparison(
        np.array([1.0]), np.array([1.0000005]), True
    )
    assert equal[0] and not exact[0]


def test_nonfinite_masks_are_checked() -> None:
    """相同 NaN/无穷通过，单端 NaN 或无穷符号不同必须失败。"""
    ref = np.array([np.nan, np.inf, -np.inf, 1.0])
    pred = np.array([np.nan, np.inf, np.inf, np.nan])
    equal, _, _, mask_bad = _module.numeric_comparison(ref, pred, True)
    np.testing.assert_array_equal(equal, [True, True, False, False])
    np.testing.assert_array_equal(mask_bad, [False, False, True, True])
