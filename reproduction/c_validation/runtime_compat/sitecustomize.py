# Copyright [2024] Expedia, Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS.
"""本机受限 Windows 令牌的临时目录兼容，不更改计算或模型逻辑。"""

import os
import pathlib  # 先初始化 pathlib，避免替换内建函数后触发方法绑定。
from typing import Optional, Union

_mkdir = os.mkdir
_task_temp = os.environ.get("KAMAE_C_TEMP_ROOT", "")


def _compatible_mkdir(
    path: Union[str, bytes, os.PathLike],
    mode: int = 0o777,
    *,
    dir_fd: Optional[int] = None,
) -> None:
    # Python 3.10 新版的 0700 ACL 会排除当前沙箱的受限 SID，导致 Spark
    # 在刚创建的临时子目录中无法创建连接文件。仅对此项目临时根生效。
    if os.name == "nt" and mode == 0o700 and _task_temp:
        absolute = os.path.abspath(os.fsdecode(path))
        if (
            os.path.splitdrive(absolute)[0].casefold()
            == os.path.splitdrive(_task_temp)[0].casefold()
            and os.path.commonpath([absolute, _task_temp]).casefold()
            == _task_temp.casefold()
        ):
            mode = 0o777
    return _mkdir(path, mode, dir_fd=dir_fd)


if os.name == "nt" and _task_temp:
    os.mkdir = _compatible_mkdir
