# Copyright [2024] Expedia, Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS.
"""用实际分桶结果展示一致性边界，完整保留全部测试输入。"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent


def main() -> None:
    """导出单面板图；没有随机重复，不添加虚构的误差条。"""
    with (HERE / "results/bucketize_cases.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    grouped = [[r for r in rows if r["dtype"] == d] for d in ["float32", "float64"]]
    assert [len(g) for g in grouped] == [16, 16]
    errors = np.array([[int(r["abs_error"]) for r in group] for group in grouped])
    labels = [
        "−2",
        "0.05",
        "0.5",
        "2",
        "0左",
        "0",
        "0右",
        "0.1左",
        "0.1",
        "0.1右",
        "1左",
        "1",
        "1右",
        "−∞",
        "+∞",
        "NaN",
    ]
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"],
            "font.size": 8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(183 / 25.4, 64 / 25.4))
    ax.imshow(
        errors != 0,
        cmap=ListedColormap(["#e6eef1", "#d69247"]),
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    ax.set_xticks(range(16), labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks([0, 1], ["float32", "float64"])
    ax.set_xlabel("测试输入（左、右表示该精度下的相邻可表示浮点数）")
    ax.set_ylabel("输入精度")
    for y, x in np.argwhere(errors != 0):
        ax.text(x, y, str(errors[y, x]), ha="center", va="center", fontsize=8)
    fig.legend(
        handles=[
            Patch(color="#e6eef1", label="两端完全一致"),
            Patch(color="#d69247", label="存在差异；格内为桶索引绝对差"),
        ],
        loc="upper center",
        ncols=2,
        frameon=False,
        fontsize=8,
    )
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.39, top=0.78)
    base = HERE / "results/bucketize_parity"
    # 单面板无需跨面板对齐；记录实际布局供复核。
    from audit_panel_alignment import require_matplotlib_panel_alignment

    require_matplotlib_panel_alignment(
        fig, json_out=str(base) + ".alignment.json", strict=True
    )
    fig.savefig(str(base) + ".pdf")
    fig.savefig(str(base) + ".svg")
    fig.savefig(str(base) + ".png", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
