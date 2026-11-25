# Copyright [2024] Expedia, Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS.
"""新进程仅加载 TensorFlow SavedModel，核对导出图可独立执行。"""

import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf


def main() -> None:
    here = Path(__file__).resolve().parent
    out = here / "results/precision_aware"
    with (here / "results/bucketize_cases.csv").open(encoding="utf-8-sig") as stream:
        cases = list(csv.DictReader(stream))
    results = []
    for dtype in ["float32", "float64"]:
        # 原件保存在 C；本机 TensorFlow 的原生文件接口使用英文工作路径。
        work = Path.home() / "Documents/Codex/work" / f"kc_precision_saved_{dtype}"
        shutil.copytree(out / f"saved_adaptive_{dtype}", work, dirs_exist_ok=True)
        model = tf.saved_model.load(str(work))
        rows = [r for r in cases if r["dtype"] == dtype]
        inputs = tf.constant([[float(r["input_repr"])] for r in rows], dtype=dtype)
        actual = model.signatures["serve"](x=inputs)["bucket"].numpy().reshape(-1)
        expected = np.array([int(r["spark_bucket"]) for r in rows])
        results.append(
            {
                "dtype": dtype,
                "rows": len(rows),
                "mismatches": int(np.sum(actual != expected)),
            }
        )
    report = {
        "fresh_process": True,
        "precision_module_imported": "precision_aware" in sys.modules,
        "kamae_imported": "kamae" in sys.modules,
        "results": results,
    }
    (out / "portable_check.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    assert not report["precision_module_imported"] and not report["kamae_imported"]
    assert all(r["mismatches"] == 0 for r in results)


if __name__ == "__main__":
    main()
