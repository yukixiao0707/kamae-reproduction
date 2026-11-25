# 使用本机已下载的编译器重建 ACL 兼容类；不修改模型计算。
$ErrorActionPreference = 'Stop'
$taskWork = Join-Path $env:USERPROFILE 'Documents\Codex\work'
$taskJava = Join-Path $taskWork 'zulu17.70.201-ca-jre17.0.21.1-win_x64\bin\java.exe'
$taskPython = Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) '.venv-c310\Scripts\python.exe'
$taskClasses = Join-Path $taskWork 'kc_fs_classes'
& $taskJava -jar (Join-Path $taskWork 'kc_ecj.jar') -17 -encoding UTF-8 -cp (Join-Path $taskWork 'kc_spark\jars\hadoop-client-api-3.5.0.jar') -d $taskClasses (Join-Path $PSScriptRoot 'runtime_compat\InheritedAclLocalFileSystem.java')
if ($LASTEXITCODE -ne 0) { throw '兼容类编译失败' }
$env:KAMAE_C_CLASS_DIR = $taskClasses
$env:KAMAE_C_JAR_TARGET = Join-Path $taskWork 'kc_spark\jars\kc_runtime_fs.jar'
& $taskPython -c 'import os,pathlib,zipfile; p=pathlib.Path(os.environ["KAMAE_C_CLASS_DIR"]); z=zipfile.ZipFile(os.environ["KAMAE_C_JAR_TARGET"],"w"); [z.write(f,f.relative_to(p).as_posix()) for f in p.rglob("*.class")]; z.close()'
if ($LASTEXITCODE -ne 0) { throw '兼容类打包失败' }
