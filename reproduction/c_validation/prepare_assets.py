# Copyright [2024] Expedia, Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""下载官方 MovieLens 与便携 Java，记录来源和哈希，不改变系统环境。"""

import concurrent.futures
import hashlib
import json
import ssl
import urllib.request
import zipfile
from pathlib import Path

import certifi

HERE = Path(__file__).resolve().parent
REPRO = HERE.parent
RUNTIME = HERE / ".runtime"
RUNTIME.mkdir(exist_ok=True)


def fetch(url: str, path: Path) -> Path:
    """下载到临时文件，完整结束后再改名。"""
    if path.exists():
        return path
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(
            context=ssl.create_default_context(cafile=certifi.where())
        ),
    )
    request = urllib.request.Request(url, headers={"User-Agent": "Kamae-C-validation"})
    temporary = path.with_suffix(path.suffix + ".partial")
    print("下载", url, flush=True)
    with opener.open(request, timeout=45) as response, temporary.open("wb") as target:
        while block := response.read(1024 * 1024):
            target.write(block)
    temporary.replace(path)
    return path


def setup_data() -> dict:
    """准备 MovieLens 原始输入。"""
    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    archive = fetch(url, RUNTIME / "ml-1m.zip")
    directory = REPRO / "data"
    directory.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        source.extractall(directory)
    return {
        "url": url,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "files": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (directory / "ml-1m").glob("*.dat")
        },
    }


def setup_java() -> dict:
    """使用包含向量模块的完整 Java 17 运行时，不改系统配置。"""
    work = Path.home() / "Documents" / "Codex" / "work"
    work.mkdir(parents=True, exist_ok=True)
    name = "zulu17.70.201-ca-jre17.0.21.1-win_x64"
    url = f"https://cdn.azul.com/zulu/bin/{name}.zip"
    archive = fetch(url, work / "zulu17.zip")
    actual = hashlib.sha256(archive.read_bytes()).hexdigest()
    expected = "7ac4b62ed405c724dfd5731442f77a655d0bfc8dd3e92921702292280aa75f55"
    if actual != expected:
        raise ValueError("Java 归档与本实验固定哈希不符")
    java_home = work / name
    if not (java_home / "bin" / "java.exe").exists():
        with zipfile.ZipFile(archive) as source:
            source.extractall(work)
    return {
        "url": url,
        "sha256": actual,
        "java_home": str(java_home),
        "version": "17.0.21.1",
        "hash_kind": "本实验归档固定 SHA256",
    }


if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        data_future = executor.submit(setup_data)
        java_future = executor.submit(setup_java)
        manifest = {"data": data_future.result(), "java": java_future.result()}
    (HERE / "assets_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
