# Copyright [2024] Expedia, Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS.
"""记录 C 的实际环境并验证 CPU、哈希和本机 Spark。"""

import faulthandler
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

faulthandler.dump_traceback_later(45, repeat=True)

import farmhash
import keras
import numpy as np
import pyspark
import tensorflow as tf
from pyspark.sql import SparkSession

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

if __name__ == "__main__":
    data = REPO / "reproduction" / "data" / "ml-1m"
    report = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "pyspark": pyspark.__version__,
        "numpy": np.__version__,
        "gpu_devices": [str(x) for x in tf.config.list_physical_devices("GPU")],
        "java_home": os.environ.get("JAVA_HOME"),
        "commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "data_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in data.glob("*.dat")
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    samples = ["", "1", "123", "中文", "Comedy", "unseen_" + "x" * 100]
    fp = tf.raw_ops.Fingerprint(data=tf.constant(samples), method="farmhash64").numpy()
    for sample, row in zip(samples, fp):
        print("核对哈希", repr(sample), flush=True)
        assert farmhash.hash64(sample) == int.from_bytes(
            row.tobytes(), "little"
        ), sample
    report["farmhash_tf_fingerprint_match"] = True
    spark = (
        SparkSession.builder.master("local[1]")
        .appName("C-environment-check")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )
    try:
        report["spark_count"] = spark.range(10).count()
        report["spark_version"] = spark.version
    finally:
        spark.stop()
    (HERE / "results" / "environment.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("环境自检通过", flush=True)
    faulthandler.cancel_dump_traceback_later()
